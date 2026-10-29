import os, glob, numpy as np
from depth_anything_3.api import DepthAnything3

MODEL_DIR = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
files = sorted(glob.glob("/workspace/data/tandt/truck/images/*"))
idx = np.linspace(0, len(files)-1, 24).round().astype(int)
ctx = [files[i] for i in idx]
print(f"pose-free 입력: {len(ctx)}장")

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=ctx,                       # 포즈 안 넘김 = pose-free
    infer_gs=True,
    process_res=480,
    process_res_method="upper_bound_resize",
    export_dir="/workspace/results/posefree_24",
    export_format="gs_ply",
)

H, W = pred.processed_images.shape[1:3]
N = pred.gaussians.means.shape[1]
print(f"\n처리: {pred.processed_images.shape[0]}장, {H}x{W}")
print(f"Gaussian: {N:,}")
print(f"예측 포즈 shape: {np.asarray(pred.extrinsics).shape}")
print(f"depth range: [{pred.depth.min():.3f}, {pred.depth.max():.3f}]")

for root,_,fs in os.walk("/workspace/results/posefree_24"):
    for f in fs:
        p = os.path.join(root,f)
        print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")