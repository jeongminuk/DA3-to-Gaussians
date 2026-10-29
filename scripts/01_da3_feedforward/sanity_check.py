import os, numpy as np, torch, imageio
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
N_CONTEXT = 12

# 1) COLMAP 로드 (sparse/0/) + 파일명 정렬
image_files, extrinsics, intrinsics = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extrinsics = np.asarray(extrinsics)[order]
intrinsics = np.asarray(intrinsics)[order]
print(f"Total: {len(image_files)} images, extr {extrinsics.shape}, intr {intrinsics.shape}")
print(f"First 3: {[os.path.basename(f) for f in image_files[:3]]}")

# 2) 균등 간격 12장
idx = np.linspace(0, len(image_files) - 1, N_CONTEXT).round().astype(int)
print(f"Context indices: {idx.tolist()}")
ctx_files = [image_files[i] for i in idx]
ctx_extr = extrinsics[idx]
ctx_intr = intrinsics[idx]

# 3) DA3 추론 (포즈 제공 + GS)
model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=ctx_files,
    extrinsics=ctx_extr,
    intrinsics=ctx_intr,
    infer_gs=True,
    align_to_input_ext_scale=True,
)
print("\n=== Prediction ===")
print("processed_images:", pred.processed_images.shape)
print("depth:", pred.depth.shape)
print("extrinsics:", pred.extrinsics.shape)
print("intrinsics:", pred.intrinsics.shape)
print("aux keys:", list(pred.aux.keys()) if hasattr(pred, "aux") else "none")
print("has .gaussians:", hasattr(pred, "gaussians"))

# 4) Gaussian 찾기
gaussians = getattr(pred, "gaussians", None)
if gaussians is None and hasattr(pred, "aux"):
    gaussians = pred.aux.get("gaussians", None)
if gaussians is None:
    raise SystemExit("Gaussians not found — check attribute name above")
print("\n=== Gaussians ===")
for name in ["means", "scales", "rotations", "opacities", "harmonics"]:
    t = getattr(gaussians, name)
    print(f"  {name}: {tuple(t.shape)}  {t.dtype}")

# 5) context view 자기 포즈로 재렌더링
H, W = pred.processed_images.shape[1:3]
print(f"\nRender resolution: {H}x{W}")

extr_t = torch.from_numpy(np.asarray(pred.extrinsics)).float().cuda()
if extr_t.shape[-2] == 3:
    bottom = torch.tensor([0., 0., 0., 1.], device="cuda").expand(extr_t.shape[0], 1, 4)
    extr_t = torch.cat([extr_t, bottom], dim=1)
intr_t = torch.from_numpy(np.asarray(pred.intrinsics)).float().cuda()

intr_n = intr_t.clone()
intr_n[..., 0, :] /= W
intr_n[..., 1, :] /= H

with torch.no_grad():
    colors, depths = render_3dgs(
        extrinsics=extr_t,
        intrinsics=intr_n,
        image_shape=(H, W),
        gaussian=gaussians,
        num_view=extr_t.shape[0],
    )
print("rendered:", colors.shape)

# 6) PSNR
gt = torch.from_numpy(pred.processed_images).float().cuda() / 255.0
rd = colors.permute(0, 2, 3, 1).clamp(0, 1)
mse = ((gt - rd) ** 2).mean(dim=(1, 2, 3))
psnr = -10 * torch.log10(mse)
print("\n=== Context-view reconstruction PSNR ===")
for i, p in enumerate(psnr):
    print(f"  view {idx[i]:3d}: {p.item():.2f} dB")
print(f"  MEAN: {psnr.mean().item():.2f} dB")

os.makedirs("/workspace/results/sanity", exist_ok=True)
for i in range(min(3, rd.shape[0])):
    imageio.imwrite(f"/workspace/results/sanity/render_{i}.png", (rd[i].cpu().numpy() * 255).astype(np.uint8))
    imageio.imwrite(f"/workspace/results/sanity/gt_{i}.png", pred.processed_images[i])
print("\nSaved to /workspace/results/sanity/")