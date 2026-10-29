"""
da3_infer.py   (da3 container)
Statue few-shot: run DA3 on train_<N> images with both pose heads and save
everything needed to build a COLMAP-format dataset (pose, K, depth, colors).

Note: Prediction.extrinsics is world-to-camera (3x4) for BOTH pose heads.
"""
import glob, json, os, time
import numpy as np, torch
from depth_anything_3.api import DepthAnything3

DATA = "/workspace/data/Statue/fs"
OUT = "/workspace/exp/Statue/fs/da3full"
MODEL = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"

model = DepthAnything3.from_pretrained(MODEL).to("cuda")
os.makedirs(OUT, exist_ok=True)
for N in (12, 24, 50):
    imgs = sorted(glob.glob(f"{DATA}/train_{N}/*.jpg"))
    for ray in (False, True):
        tag = f"{N}_{'ray' if ray else 'decoder'}"
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pred = model.inference(image=imgs, use_ray_pose=ray,
                               ref_view_strategy="saddle_balanced", process_res=504)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        np.savez(f"{OUT}/{tag}.npz",
                 extrinsics=np.asarray(pred.extrinsics), intrinsics=np.asarray(pred.intrinsics),
                 depth=pred.depth.astype(np.float32), conf=pred.conf.astype(np.float32),
                 images=pred.processed_images.astype(np.uint8),
                 names=np.array([os.path.basename(p) for p in imgs]), seconds=dt)
        print(tag, f"{dt:.2f}s", pred.depth.shape, flush=True)
