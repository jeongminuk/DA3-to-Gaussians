import os, numpy as np, torch, imageio
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
N_CONTEXT = 12

image_files, extrinsics, intrinsics = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extrinsics = np.asarray(extrinsics)[order]
intrinsics = np.asarray(intrinsics)[order]

idx = np.linspace(0, len(image_files) - 1, N_CONTEXT).round().astype(int)
ctx_files = [image_files[i] for i in idx]
ctx_extr = extrinsics[idx]
ctx_intr = intrinsics[idx]

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")

# 핵심 변경: align_to_input_ext_scale=False -> Gaussian과 같은 좌표계의 포즈를 받음
pred = model.inference(
    image=ctx_files,
    extrinsics=ctx_extr,
    intrinsics=ctx_intr,
    infer_gs=True,
    align_to_input_ext_scale=False,
)

gaussians = pred.gaussians
H, W = pred.processed_images.shape[1:3]

# 좌표계가 실제로 다른지 수치로 확인
pe = np.asarray(pred.extrinsics)
if pe.shape[-2] == 3:
    pe4 = np.concatenate([pe, np.tile(np.array([[[0,0,0,1.]]]), (pe.shape[0],1,1))], axis=1)
else:
    pe4 = pe
print("=== Pose comparison (DA3 internal vs COLMAP) ===")
print("R diff (mean abs):", np.abs(pe4[:,:3,:3] - ctx_extr[:,:3,:3]).mean())
print("t diff (mean abs):", np.abs(pe4[:,:3,3] - ctx_extr[:,:3,3]).mean())
print("DA3 t[0]:", pe4[0,:3,3])
print("COLMAP t[0]:", ctx_extr[0,:3,3])

extr_t = torch.from_numpy(pe4).float().cuda()
intr_t = torch.from_numpy(np.asarray(pred.intrinsics)).float().cuda()
intr_n = intr_t.clone()
intr_n[..., 0, :] /= W
intr_n[..., 1, :] /= H

with torch.no_grad():
    colors, _ = render_3dgs(
        extrinsics=extr_t, intrinsics=intr_n,
        image_shape=(H, W), gaussian=gaussians,
        num_view=extr_t.shape[0],
    )

gt = torch.from_numpy(pred.processed_images).float().cuda() / 255.0
rd = colors.permute(0, 2, 3, 1).clamp(0, 1)
psnr = -10 * torch.log10(((gt - rd) ** 2).mean(dim=(1,2,3)))
print("\n=== PSNR with DA3 internal poses ===")
for i, p in enumerate(psnr):
    print(f"  view {idx[i]:3d}: {p.item():.2f} dB")
print(f"  MEAN: {psnr.mean().item():.2f} dB")

os.makedirs("/workspace/results/sanity2", exist_ok=True)
for i in range(3):
    imageio.imwrite(f"/workspace/results/sanity2/render_{i}.png", (rd[i].cpu().numpy()*255).astype(np.uint8))
    imageio.imwrite(f"/workspace/results/sanity2/gt_{i}.png", pred.processed_images[i])
print("Saved to /workspace/results/sanity2/")