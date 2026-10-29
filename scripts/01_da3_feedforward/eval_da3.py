import os, numpy as np, torch, imageio
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR  = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
OUT_DIR    = "/workspace/results/eval_da3"
N_CONTEXT  = 12
TEST_EVERY = 8
LPIPS_NET  = "alex"          # gsplat 기본값과 통일
CONTEXT_MODE = "uniform"     # "uniform" | "narrow"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1) COLMAP 로드 (파일명 정렬) ----------
image_files, extr_all, intr_all = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extr_all = np.asarray(extr_all)[order]      # w2c (N,4,4)
intr_all = np.asarray(intr_all)[order]
n_total = len(image_files)

# ---------- 2) test / context 분리 ----------
test_idx = np.arange(0, n_total, TEST_EVERY)                  # gsplat과 동일 규칙
train_pool = np.setdiff1d(np.arange(n_total), test_idx)       # test 제외한 나머지
if CONTEXT_MODE == "uniform":
    sel = np.linspace(0, len(train_pool) - 1, N_CONTEXT).round().astype(int)
else:
    sel = np.arange(N_CONTEXT) * 2
ctx_idx = train_pool[sel]

print(f"total {n_total} | context {len(ctx_idx)} {ctx_idx.tolist()}")
print(f"test {len(test_idx)} views (every {TEST_EVERY})")

# ---------- 3) DA3 추론 ----------
model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=[image_files[i] for i in ctx_idx],
    extrinsics=extr_all[ctx_idx],
    intrinsics=intr_all[ctx_idx],
    infer_gs=True,
    align_to_input_ext_scale=False,     # Gaussian과 같은 좌표계 유지
)
g = pred.gaussians
H, W = pred.processed_images.shape[1:3]
print(f"render res {H}x{W} | N_gaussians {g.means.shape[1]:,}")

def to4x4(e):
    e = np.asarray(e)
    if e.shape[-2] == 3:
        e = np.concatenate([e, np.tile(np.array([[[0,0,0,1.]]]), (e.shape[0],1,1))], axis=1)
    return e

da3_extr = to4x4(pred.extrinsics)          # DA3 좌표계 w2c
col_extr = to4x4(extr_all[ctx_idx])        # COLMAP 좌표계 w2c

# ---------- 4) Sim(3): COLMAP -> DA3 (카메라 중심 기반 Umeyama) ----------
def centers(E):
    R, t = E[:, :3, :3], E[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0,2,1), -t)

def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: F[2,2] = -1
    R = U @ F @ Vt
    s = (sig * np.diag(F)).sum() / (S**2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t

s_a, R_a, t_a = umeyama(centers(col_extr), centers(da3_extr))
print(f"Sim3: scale={s_a:.5f}")
resid = np.linalg.norm(
    (s_a * (R_a @ centers(col_extr).T).T + t_a) - centers(da3_extr), axis=1)
print(f"alignment residual: mean {resid.mean():.5f}, max {resid.max():.5f}")

def colmap_to_da3(E):
    E = to4x4(E)
    Rd = E[:, :3, :3] @ R_a.T
    td = s_a * E[:, :3, 3] - np.einsum("nij,j->ni", Rd, t_a)
    out = np.tile(np.eye(4), (len(E),1,1))
    out[:, :3, :3], out[:, :3, 3] = Rd, td
    return out

# ---------- 5) GT 이미지 (DA3 처리 해상도에 맞춤) ----------
def load_gt(paths):
    arr = [np.asarray(Image.open(p).convert("RGB").resize((W,H), Image.LANCZOS)) for p in paths]
    return torch.from_numpy(np.stack(arr)).float().cuda() / 255.0

# ---------- 6) metric (gsplat simple_trainer.py와 동일 구현) ----------
psnr_fn  = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type=LPIPS_NET, normalize=True).cuda()

def evaluate(idxs, tag, chunk=4):
    extr_t = torch.from_numpy(colmap_to_da3(extr_all[idxs])).float().cuda()
    K = torch.from_numpy(np.asarray(pred.intrinsics)[0]).float().cuda()
    K = K.unsqueeze(0).repeat(len(idxs), 1, 1)
    Kn = K.clone(); Kn[:,0,:] /= W; Kn[:,1,:] /= H
    gt = load_gt([image_files[i] for i in idxs])

    outs = []
    with torch.no_grad():
        for s in range(0, len(idxs), chunk):
            e = extr_t[s:s+chunk]
            c, _ = render_3dgs(extrinsics=e, intrinsics=Kn[s:s+chunk],
                               image_shape=(H,W), gaussian=g, num_view=len(e))
            outs.append(c)
    rd = torch.cat(outs).permute(0,2,3,1).clamp(0,1)

    P, S, L = [], [], []
    for i in range(len(idxs)):
        a = rd[i:i+1].permute(0,3,1,2)
        b = gt[i:i+1].permute(0,3,1,2)
        P.append(psnr_fn(a,b).item()); S.append(ssim_fn(a,b).item()); L.append(lpips_fn(a,b).item())

    print(f"\n=== {tag} ({len(idxs)} views) ===")
    print(f"PSNR  {np.mean(P):.3f}")
    print(f"SSIM  {np.mean(S):.4f}")
    print(f"LPIPS {np.mean(L):.4f}  (net={LPIPS_NET})")
    for i in range(min(3, len(idxs))):
        imageio.imwrite(f"{OUT_DIR}/{tag}_render_{i}.png", (rd[i].cpu().numpy()*255).astype(np.uint8))
        imageio.imwrite(f"{OUT_DIR}/{tag}_gt_{i}.png", (gt[i].cpu().numpy()*255).astype(np.uint8))
    return np.mean(P), np.mean(S), np.mean(L)

ctx_res  = evaluate(ctx_idx,  "context")   # sanity: 본 뷰
test_res = evaluate(test_idx, "test")      # 진짜 NVS

print("\n" + "="*46)
print(f"{'':8s}{'PSNR':>9s}{'SSIM':>9s}{'LPIPS':>9s}")
print(f"{'context':8s}{ctx_res[0]:9.3f}{ctx_res[1]:9.4f}{ctx_res[2]:9.4f}")
print(f"{'test':8s}{test_res[0]:9.3f}{test_res[1]:9.4f}{test_res[2]:9.4f}")
print(f"\nN_gaussians: {g.means.shape[1]:,}")
print(f"Saved to {OUT_DIR}/")