import os, gc, numpy as np, torch, time, json
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR  = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
TEST_EVERY = 8
PROCESS_RES = 480
CONTEXT_COUNTS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 180, 219]  # 독립변수
REF_STRATEGY = "saddle_balanced"
LPIPS_NET = "alex"

# --- COLMAP 로드 + 파일명 정렬 ---
image_files, extr, intr = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extr, intr = np.asarray(extr)[order], np.asarray(intr)[order]
n = len(image_files)

# --- test / train pool 고정 분할 ---
test_idx = np.arange(0, n, TEST_EVERY)
train_pool = np.setdiff1d(np.arange(n), test_idx)
print(f"total {n} | test {len(test_idx)} (fixed) | train pool {len(train_pool)}")

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type=LPIPS_NET, normalize=True).cuda()

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
    s = (sig*np.diag(F)).sum() / (S**2).sum() * len(src)
    return s, R, mu_d - s*R@mu_s

def load_gt(paths, H, W):
    a = [np.asarray(Image.open(p).convert("RGB").resize((W,H), Image.LANCZOS)) for p in paths]
    return torch.from_numpy(np.stack(a)).float().cuda()/255.0

results = []
for V in CONTEXT_COUNTS:
    if V > len(train_pool):
        continue
    ctx_idx = train_pool[np.linspace(0, len(train_pool)-1, V).round().astype(int)]

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); gc.collect()
    row = {"context_views": V}
    try:
        t0 = time.time()
        pred = model.inference(
            image=[image_files[i] for i in ctx_idx],
            extrinsics=extr[ctx_idx], intrinsics=intr[ctx_idx],
            infer_gs=True, align_to_input_ext_scale=False,
            ref_view_strategy=REF_STRATEGY,
            process_res=PROCESS_RES, process_res_method="upper_bound_resize",
        )
        torch.cuda.synchronize()
        infer_t = time.time() - t0
        g = pred.gaussians
        H, W = pred.processed_images.shape[1:3]

        # Sim(3): COLMAP context -> DA3 context
        da3_c = to4x4(pred.extrinsics)
        col_c = to4x4(extr[ctx_idx])
        s_a, R_a, t_a = umeyama(centers(col_c), centers(da3_c))

        def col2da3(E):
            E = to4x4(E)
            Rd = E[:,:3,:3] @ R_a.T
            td = s_a*E[:,:3,3] - np.einsum("nij,j->ni", Rd, t_a)
            out = np.tile(np.eye(4), (len(E),1,1))
            out[:,:3,:3], out[:,:3,3] = Rd, td
            return out

        # test 뷰 렌더링
        te = torch.from_numpy(col2da3(extr[test_idx])).float().cuda()
        K = torch.from_numpy(np.asarray(pred.intrinsics)[0]).float().cuda()
        K = K.unsqueeze(0).repeat(len(test_idx),1,1)
        Kn = K.clone(); Kn[:,0,:]/=W; Kn[:,1,:]/=H
        gt = load_gt([image_files[i] for i in test_idx], H, W)

        outs = []
        with torch.no_grad():
            for s in range(0, len(test_idx), 4):
                c,_ = render_3dgs(extrinsics=te[s:s+4], intrinsics=Kn[s:s+4],
                                  image_shape=(H,W), gaussian=g, num_view=len(te[s:s+4]))
                outs.append(c)
        rd = torch.cat(outs).permute(0,2,3,1).clamp(0,1)

        P,S,L = [],[],[]
        for i in range(len(test_idx)):
            a = rd[i:i+1].permute(0,3,1,2); b = gt[i:i+1].permute(0,3,1,2)
            P.append(psnr_fn(a,b).item()); S.append(ssim_fn(a,b).item()); L.append(lpips_fn(a,b).item())

        row.update({
            "H": H, "W": W, "n_gaussians": g.means.shape[1],
            "peak_vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
            "infer_time_s": round(infer_t, 2),
            "PSNR": round(np.mean(P), 3), "SSIM": round(np.mean(S), 4),
            "LPIPS": round(np.mean(L), 4), "status": "ok",
        })
        print(f"[ok] ctx={V}: PSNR {row['PSNR']}, VRAM {row['peak_vram_gb']}GB, "
              f"{row['n_gaussians']:,} GS")
        del pred, g
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"; print(f"[OOM] ctx={V}")
    except Exception as e:
        row["status"] = f"ERR: {type(e).__name__}: {str(e)[:100]}"
        print(f"[ERR] ctx={V}: {row['status']}")
    results.append(row)
    torch.cuda.empty_cache(); gc.collect()

os.makedirs("/workspace/results/view_scaling_eval", exist_ok=True)
with open("/workspace/results/view_scaling_eval/results.json","w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*88)
print(f"{'ctx':>5}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'#GS':>12}{'VRAM':>9}{'time':>8}{'status':>8}")
for r in results:
    if r["status"]=="ok":
        print(f"{r['context_views']:>5}{r['PSNR']:>8}{r['SSIM']:>8}{r['LPIPS']:>8}"
              f"{r['n_gaussians']:>12,}{r['peak_vram_gb']:>9}{r['infer_time_s']:>8}{'ok':>8}")
    else:
        print(f"{r['context_views']:>5}{'—':>8}{'—':>8}{'—':>8}{'—':>12}{'—':>9}{'—':>8}{r['status']:>8}")