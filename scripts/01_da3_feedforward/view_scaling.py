import os, gc, numpy as np, torch, time, json
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR  = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
PROCESS_RES = 480                      # 270x480 근사 (긴 변 480)
VIEW_COUNTS = [2, 4, 8, 12, 16, 24, 32, 48, 64]   # 독립변수
REF_STRATEGY = "saddle_balanced"       # 고정

image_files, extr, intr = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extr, intr = np.asarray(extr)[order], np.asarray(intr)[order]
n_total = len(image_files)

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")

results = []
for V in VIEW_COUNTS:
    if V > n_total:
        print(f"[skip] {V} > {n_total}")
        continue

    # 균등 샘플링 (모든 V에서 동일 로직)
    idx = np.linspace(0, n_total - 1, V).round().astype(int)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); gc.collect()

    row = {"views_requested": V}
    try:
        t0 = time.time()
        pred = model.inference(
            image=[image_files[i] for i in idx],
            extrinsics=extr[idx], intrinsics=intr[idx],
            infer_gs=True,
            align_to_input_ext_scale=False,     # 고정
            ref_view_strategy=REF_STRATEGY,     # 고정
            process_res=PROCESS_RES,            # 고정
            process_res_method="upper_bound_resize",
        )
        torch.cuda.synchronize()
        dt = time.time() - t0

        H, W = pred.processed_images.shape[1:3]
        N = pred.gaussians.means.shape[1]
        row.update({
            "views_processed": pred.processed_images.shape[0],
            "H": H, "W": W,
            "n_gaussians": N,
            "gaussians_per_view": N // pred.processed_images.shape[0],
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "wall_time_s": round(dt, 2),
            "status": "ok",
        })
        print(f"[ok] V={V}: {pred.processed_images.shape[0]} views, "
              f"{H}x{W}, {N:,} GS, {row['peak_vram_gb']} GB, {dt:.1f}s")
        del pred
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
        print(f"[OOM] V={V}")
    except Exception as e:
        row["status"] = f"ERROR: {type(e).__name__}: {str(e)[:120]}"
        print(f"[ERR] V={V}: {row['status']}")

    results.append(row)
    torch.cuda.empty_cache(); gc.collect()

os.makedirs("/workspace/results/view_scaling", exist_ok=True)
with open("/workspace/results/view_scaling/results.json", "w") as f:
    json.dump(results, f, indent=2)

# 요약 표
print("\n" + "="*80)
print(f"{'V_req':>6}{'V_proc':>7}{'HxW':>12}{'#GS':>12}{'GS/view':>10}{'VRAM(GB)':>10}{'time(s)':>9}{'status':>8}")
for r in results:
    if r["status"] == "ok":
        print(f"{r['views_requested']:>6}{r['views_processed']:>7}"
              f"{r['H']}x{r['W']:>7}{r['n_gaussians']:>12,}{r['gaussians_per_view']:>10,}"
              f"{r['peak_vram_gb']:>10}{r['wall_time_s']:>9}{'ok':>8}")
    else:
        print(f"{r['views_requested']:>6}{'—':>7}{'—':>12}{'—':>12}{'—':>10}{'—':>10}{'—':>9}{r['status']:>8}")
print("\nSaved to /workspace/results/view_scaling/results.json")