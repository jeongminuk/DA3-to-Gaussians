import os, glob, numpy as np, torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler

image_files, extr, intr = ColmapHandler.process("/workspace/data/tandt/truck", "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extr, intr = np.asarray(extr)[order], np.asarray(intr)[order]

idx = np.linspace(0, len(image_files)-1, 24).round().astype(int)   # 24장
print(f"입력 요청: {len(idx)}장")

model = DepthAnything3.from_pretrained("/workspace/models/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
pred = model.inference(
    image=[image_files[i] for i in idx],
    extrinsics=extr[idx], intrinsics=intr[idx],
    infer_gs=True, align_to_input_ext_scale=False,
)

print(f"처리된 이미지: {pred.processed_images.shape}")   # 첫 차원이 실제 뷰 수
print(f"depth: {pred.depth.shape}")
print(f"extrinsics 반환: {np.asarray(pred.extrinsics).shape}")
N = pred.gaussians.means.shape[1]
H, W = pred.processed_images.shape[1:3]
print(f"Gaussian: {N:,} = {N//(H*W)} views × {H} × {W}")