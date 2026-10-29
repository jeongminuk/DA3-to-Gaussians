import numpy as np, torch, os
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler

image_files, extr, intr = ColmapHandler.process("/workspace/data/tandt/truck", "0")
o = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in o]
extr = np.asarray(extr)[o]; intr = np.asarray(intr)[o]

model = DepthAnything3.from_pretrained("/workspace/models/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
pred = model.inference(image=image_files, infer_gs=False,
                       process_res=480, process_res_method="upper_bound_resize")

H, W = pred.processed_images.shape[1:3]
print(f"{pred.processed_images.shape[0]}장 | {H}x{W} | 총 픽셀 {251*H*W:,}")
print(f"depth {pred.depth.shape} | conf {pred.conf.shape}")

np.savez("/workspace/results/da3_geometry.npz",
    depth=pred.depth.astype(np.float32),
    conf=pred.conf.astype(np.float32),
    images=pred.processed_images.astype(np.uint8),
    extrinsics=np.asarray(pred.extrinsics),
    intrinsics=np.asarray(pred.intrinsics),
    colmap_extrinsics=extr,
    names=np.array([os.path.basename(f) for f in image_files]))
print("saved: /workspace/results/da3_geometry.npz")