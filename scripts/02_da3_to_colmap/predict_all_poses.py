import numpy as np, torch, os, time
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler

image_files, extr, intr = ColmapHandler.process("/workspace/data/tandt/truck", "0")
o = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in o]
extr = np.asarray(extr)[o]; intr = np.asarray(intr)[o]
print(f"입력: {len(image_files)}장")

model = DepthAnything3.from_pretrained("/workspace/models/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
pred = model.inference(
    image=image_files,
    infer_gs=False,                      # GS head 끔 → 메모리 대폭 절약
    process_res=480, process_res_method="upper_bound_resize",
)
torch.cuda.synchronize()
dt = time.time() - t0

print(f"처리 {pred.processed_images.shape[0]}장 | {dt:.1f}s | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"extrinsics {np.asarray(pred.extrinsics).shape}")

os.makedirs("/workspace/results", exist_ok=True)
np.savez("/workspace/results/da3_poses_all.npz",
         extrinsics=np.asarray(pred.extrinsics),
         intrinsics=np.asarray(pred.intrinsics),
         colmap_extrinsics=extr,
         names=np.array([os.path.basename(f) for f in image_files]))
print("saved: /workspace/results/da3_poses_all.npz")