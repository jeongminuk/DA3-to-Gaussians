import os, numpy as np, torch, imageio
from types import SimpleNamespace
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"

image_files, extrinsics, intrinsics = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extrinsics = np.asarray(extrinsics)[order]
intrinsics = np.asarray(intrinsics)[order]
idx = np.linspace(0, len(image_files) - 1, 12).round().astype(int)

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=[image_files[i] for i in idx],
    extrinsics=extrinsics[idx], intrinsics=intrinsics[idx],
    infer_gs=True, align_to_input_ext_scale=False,
)

# [1] Prediction에 뭐가 들어있는지 전부 확인
print("=== pred attributes ===")
for k in dir(pred):
    if k.startswith("_"): continue
    v = getattr(pred, k, None)
    if callable(v): continue
    if isinstance(v, np.ndarray): print(f"  {k}: ndarray{v.shape}")
    elif isinstance(v, (int, float)): print(f"  {k}: {v}")
    else: print(f"  {k}: {type(v).__name__}")

g = pred.gaussians
H, W = pred.processed_images.shape[1:3]
N_per_view = H * W
print(f"\nH={H} W={W}, per-view N={N_per_view}, total N={g.means.shape[1]}")

extr = np.asarray(pred.extrinsics)
if extr.shape[-2] == 3:
    extr = np.concatenate([extr, np.tile(np.array([[[0,0,0,1.]]]), (extr.shape[0],1,1))], axis=1)
extr_t = torch.from_numpy(extr).float().cuda()
intr_t = torch.from_numpy(np.asarray(pred.intrinsics)).float().cuda()
intr_n = intr_t.clone(); intr_n[...,0,:] /= W; intr_n[...,1,:] /= H
gt = torch.from_numpy(pred.processed_images).float().cuda() / 255.0

def psnr_of(colors, gt_img):
    rd = colors.permute(0,2,3,1).clamp(0,1)
    return (-10*torch.log10(((gt_img - rd)**2).mean(dim=(1,2,3)))), rd

# [2] 뷰 0의 Gaussian만으로 뷰 0 렌더링 (교차뷰 정합 문제 배제)
g0 = SimpleNamespace(
    means=g.means[:, :N_per_view], scales=g.scales[:, :N_per_view],
    rotations=g.rotations[:, :N_per_view], opacities=g.opacities[:, :N_per_view],
    harmonics=g.harmonics[:, :N_per_view],
)
with torch.no_grad():
    c_single, _ = render_3dgs(extrinsics=extr_t[:1], intrinsics=intr_n[:1],
                              image_shape=(H,W), gaussian=g0, num_view=1)
p, rd = psnr_of(c_single, gt[:1])
print(f"\n[A] single-view (view0 GS -> view0 pose): {p.item():.2f} dB")
os.makedirs("/workspace/results/diag", exist_ok=True)
imageio.imwrite("/workspace/results/diag/A_single.png", (rd[0].cpu().numpy()*255).astype(np.uint8))

# [3] scale_factor 보정 후 전체 렌더링
sf = getattr(pred, "scale_factor", None)
print(f"\nscale_factor = {sf}")
if sf is not None:
    for name, factor in [("divide", 1.0/sf), ("multiply", sf)]:
        e = extr_t.clone(); e[:, :3, 3] *= factor
        with torch.no_grad():
            c, _ = render_3dgs(extrinsics=e, intrinsics=intr_n,
                               image_shape=(H,W), gaussian=g, num_view=12)
        p, rd = psnr_of(c, gt)
        print(f"[B] all views, t {name} by scale_factor: MEAN {p.mean().item():.2f} dB")
        imageio.imwrite(f"/workspace/results/diag/B_{name}.png",
                        (rd[0].cpu().numpy()*255).astype(np.uint8))
print("\nSaved to /workspace/results/diag/")