import os, numpy as np, torch, imageio
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

COLMAP_DIR = "/workspace/data/tandt/train"      # truck으로 바꾸면 Truck
MODEL_DIR  = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
OUT        = "/workspace/results/eval_fps_train"
N_CTX, TEST_EVERY, PRES = 12, 8, 480
os.makedirs(OUT, exist_ok=True)

# 1) COLMAP 로드 + 파일명 정렬
image_files, extr, intr = ColmapHandler.process(COLMAP_DIR, "0")
o = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in o]
extr, intr = np.asarray(extr)[o], np.asarray(intr)[o]
n = len(image_files)

def to4(e):
    e = np.asarray(e, dtype=np.float64)
    return e if e.shape[-2] == 4 else np.concatenate(
        [e, np.tile(np.array([[[0,0,0,1.]]]), (e.shape[0],1,1))], axis=1)
def cen(E):
    R, t = E[:,:3,:3], E[:,:3,3]
    return np.einsum("nij,nj->ni", R.transpose(0,2,1), -t)

# 2) test 제외 → pool
test_idx = np.arange(0, n, TEST_EVERY)
pool = np.setdiff1d(np.arange(n), test_idx)

# 3) pool에서 FPS로 12장 (카메라 위치 기준, 시작점 0 고정)
pool_c = cen(to4(extr[pool]))
sel = [0]
dist = np.linalg.norm(pool_c - pool_c[0], axis=1)
for _ in range(N_CTX - 1):
    nxt = int(np.argmax(dist))
    sel.append(nxt)
    dist = np.minimum(dist, np.linalg.norm(pool_c - pool_c[nxt], axis=1))
ctx_idx = pool[np.array(sel)]
print(f"total {n} | test {len(test_idx)} | pool {len(pool)}")
print(f"FPS context: {sorted(ctx_idx.tolist())}")

# 4) DA3 추론 (GS head ON, 포즈 제공)
model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=[image_files[i] for i in ctx_idx],
    extrinsics=extr[ctx_idx], intrinsics=intr[ctx_idx],
    infer_gs=True, align_to_input_ext_scale=False,
    process_res=PRES, process_res_method="upper_bound_resize",
)
g = pred.gaussians
H, W = pred.processed_images.shape[1:3]
print(f"render {H}x{W} | GS {g.means.shape[1]:,}")

# 5) Sim(3): COLMAP -> DA3
def umeyama(s, d):
    ms, md = s.mean(0), d.mean(0); S, D = s-ms, d-md
    C = D.T@S/len(s); U, sig, Vt = np.linalg.svd(C); F = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: F[2,2] = -1
    R = U@F@Vt; sc = (sig*np.diag(F)).sum()/(S**2).sum()*len(s)
    return sc, R, md - sc*R@ms

da3_c, col_c = to4(pred.extrinsics), to4(extr[ctx_idx])
sc, Ra, ta = umeyama(cen(col_c), cen(da3_c))
res = np.linalg.norm((sc*(Ra@cen(col_c).T).T + ta) - cen(da3_c), axis=1)
print(f"Sim3 scale {sc:.5f} | residual mean {res.mean():.5f} max {res.max():.5f}")

def c2d(E):
    E = to4(E); Rd = E[:,:3,:3]@Ra.T
    td = sc*E[:,:3,3] - np.einsum("nij,j->ni", Rd, ta)
    out = np.tile(np.eye(4), (len(E),1,1)); out[:,:3,:3], out[:,:3,3] = Rd, td
    return out

# 6) 렌더링 + metric
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()

def ev(idxs, tag):
    te = torch.from_numpy(c2d(extr[idxs])).float().cuda()
    K = torch.from_numpy(np.asarray(pred.intrinsics)[0]).float().cuda().unsqueeze(0).repeat(len(idxs),1,1)
    Kn = K.clone(); Kn[:,0,:] /= W; Kn[:,1,:] /= H
    gt = torch.from_numpy(np.stack([
        np.asarray(Image.open(image_files[i]).convert("RGB").resize((W,H), Image.LANCZOS))
        for i in idxs])).float().cuda()/255.
    outs = []
    with torch.no_grad():
        for s in range(0, len(idxs), 4):
            c, _ = render_3dgs(extrinsics=te[s:s+4], intrinsics=Kn[s:s+4],
                               image_shape=(H,W), gaussian=g, num_view=len(te[s:s+4]))
            outs.append(c)
    rd = torch.cat(outs).permute(0,2,3,1).clamp(0,1)
    P,S,L = [],[],[]
    for i in range(len(idxs)):
        a = rd[i:i+1].permute(0,3,1,2); b = gt[i:i+1].permute(0,3,1,2)
        P.append(psnr_fn(a,b).item()); S.append(ssim_fn(a,b).item()); L.append(lpips_fn(a,b).item())
    for i in range(min(3, len(idxs))):
        imageio.imwrite(f"{OUT}/{tag}_render_{i}.png", (rd[i].cpu().numpy()*255).astype(np.uint8))
        imageio.imwrite(f"{OUT}/{tag}_gt_{i}.png", (gt[i].cpu().numpy()*255).astype(np.uint8))
    print(f"{tag:8s} PSNR {np.mean(P):7.3f}  SSIM {np.mean(S):.4f}  LPIPS {np.mean(L):.4f}")
    return np.mean(P), np.mean(S), np.mean(L)

print()
ev(ctx_idx, "context")
ev(test_idx, "test")
print(f"\nGS {g.means.shape[1]:,} | context: FPS {N_CTX} | saved {OUT}/")