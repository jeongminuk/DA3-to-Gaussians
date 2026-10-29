import os, numpy as np, torch, imageio
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.model.reference_view_selector import ReferenceViewSelector
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR  = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
OUT_DIR    = "/workspace/results/eval_da3_fps"
N_CONTEXT  = 12
TEST_EVERY = 8
PROCESS_RES = 480
LPIPS_NET  = "alex"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) COLMAP 로드 + 파일명 정렬
image_files, extr, intr = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extr, intr = np.asarray(extr)[order], np.asarray(intr)[order]
n = len(image_files)

# 2) test 제외 → pool
test_idx = np.arange(0, n, TEST_EVERY)
train_pool = np.setdiff1d(np.arange(n), test_idx)
print(f"total {n} | test {len(test_idx)} | pool {len(train_pool)}")

# 3) pool에서 FPS로 12개 (논문 구현 그대로 사용)
selector = ReferenceViewSelector()
pool_extr = extr[train_pool]                                  # w2c (pool,4,4)
positions = selector._compute_camera_positions(
    torch.from_numpy(pool_extr).float()
)
fps_local = selector._fps_sampling(positions, N_CONTEXT, start_idx=0)
fps_local = np.asarray(fps_local)
ctx_idx = train_pool[fps_local]
print(f"FPS context {N_CONTEXT}: {sorted(ctx_idx.tolist())}")

# 4) DA3 추론
model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=[image_files[i] for i in ctx_idx],
    extrinsics=extr[ctx_idx], intrinsics=intr[ctx_idx],
    infer_gs=True, align_to_input_ext_scale=False,
    process_res=PROCESS_RES, process_res_method="upper_bound_resize",
)
g = pred.gaussians
H, W = pred.processed_images.shape[1:3]
print(f"render {H}x{W} | N_gaussians {g.means.shape[1]:,}")

# --- 이하 eval_da3.py와 동일 ---
def to4x4(e):
    e = np.asarray(e)
    if e.shape[-2] == 3:
        e = np.concatenate([e, np.tile(np.eye(4)[None,-1:], (e.shape[0],1,1))], axis=1)
    return e
def centers(E):
    R, t = E[:,:3,:3], E[:,:3,3]
    return np.einsum("nij,nj->ni", R.transpose(0,2,1), -t)
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src-mu_s, dst-mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    F = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: F[2,2] = -1
    R = U @ F @ Vt
    s = (sig*np.diag(F)).sum()/(S**2).sum()*len(src)
    return s, R, mu_d - s*R@mu_s

da3_c = to4x4(pred.extrinsics); col_c = to4x4(extr[ctx_idx])
s_a, R_a, t_a = umeyama(centers(col_c), centers(da3_c))
resid = np.linalg.norm((s_a*(R_a@centers(col_c).T).T + t_a) - centers(da3_c), axis=1)
print(f"Sim3 scale={s_a:.5f}, residual mean {resid.mean():.5f} max {resid.max():.5f}")

def col2da3(E):
    E = to4x4(E)
    Rd = E[:,:3,:3] @ R_a.T
    td = s_a*E[:,:3,3] - np.einsum("nij,j->ni", Rd, t_a)
    out = np.tile(np.eye(4), (len(E),1,1)); out[:,:3,:3], out[:,:3,3] = Rd, td
    return out
def load_gt(paths):
    a = [np.asarray(Image.open(p).convert("RGB").resize((W,H), Image.LANCZOS)) for p in paths]
    return torch.from_numpy(np.stack(a)).float().cuda()/255.0

psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type=LPIPS_NET, normalize=True).cuda()

def evaluate(idxs, tag):
    te = torch.from_numpy(col2da3(extr[idxs])).float().cuda()
    K = torch.from_numpy(np.asarray(pred.intrinsics)[0]).float().cuda().unsqueeze(0).repeat(len(idxs),1,1)
    Kn = K.clone(); Kn[:,0,:]/=W; Kn[:,1,:]/=H
    gt = load_gt([image_files[i] for i in idxs])
    outs=[]
    with torch.no_grad():
        for s in range(0,len(idxs),4):
            c,_ = render_3dgs(extrinsics=te[s:s+4], intrinsics=Kn[s:s+4],
                              image_shape=(H,W), gaussian=g, num_view=len(te[s:s+4]))
            outs.append(c)
    rd = torch.cat(outs).permute(0,2,3,1).clamp(0,1)
    P,S,L=[],[],[]
    for i in range(len(idxs)):
        a=rd[i:i+1].permute(0,3,1,2); b=gt[i:i+1].permute(0,3,1,2)
        P.append(psnr_fn(a,b).item()); S.append(ssim_fn(a,b).item()); L.append(lpips_fn(a,b).item())
    for i in range(min(3,len(idxs))):
        imageio.imwrite(f"{OUT_DIR}/{tag}_render_{i}.png",(rd[i].cpu().numpy()*255).astype(np.uint8))
        imageio.imwrite(f"{OUT_DIR}/{tag}_gt_{i}.png",(gt[i].cpu().numpy()*255).astype(np.uint8))
    print(f"\n=== {tag} ({len(idxs)}) ===  PSNR {np.mean(P):.3f}  SSIM {np.mean(S):.4f}  LPIPS {np.mean(L):.4f}")
    return np.mean(P), np.mean(S), np.mean(L)

c_res = evaluate(ctx_idx, "context")
t_res = evaluate(test_idx, "test")
print("\n" + "="*46)
print(f"{'':8s}{'PSNR':>9s}{'SSIM':>9s}{'LPIPS':>9s}")
print(f"{'context':8s}{c_res[0]:9.3f}{c_res[1]:9.4f}{c_res[2]:9.4f}")
print(f"{'test':8s}{t_res[0]:9.3f}{t_res[1]:9.4f}{t_res[2]:9.4f}")
print(f"\nN_gaussians: {g.means.shape[1]:,} | context selection: FPS") 