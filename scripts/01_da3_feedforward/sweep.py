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

g = pred.gaussians
H, W = pred.processed_images.shape[1:3]
N = H * W
g0 = SimpleNamespace(means=g.means[:, :N], scales=g.scales[:, :N],
                     rotations=g.rotations[:, :N], opacities=g.opacities[:, :N],
                     harmonics=g.harmonics[:, :N])

extr = np.asarray(pred.extrinsics)
if extr.shape[-2] == 3:
    extr = np.concatenate([extr, np.tile(np.array([[[0,0,0,1.]]]), (extr.shape[0],1,1))], axis=1)
extr_t = torch.from_numpy(extr).float().cuda()
intr_t = torch.from_numpy(np.asarray(pred.intrinsics)).float().cuda()
intr_n = intr_t.clone(); intr_n[...,0,:] /= W; intr_n[...,1,:] /= H
gt = torch.from_numpy(pred.processed_images).float().cuda() / 255.0

def render_psnr(gauss, e, i_n, gt_img, nv):
    with torch.no_grad():
        c, _ = render_3dgs(extrinsics=e, intrinsics=i_n, image_shape=(H,W),
                           gaussian=gauss, num_view=nv)
    rd = c.permute(0,2,3,1).clamp(0,1)
    return (-10*torch.log10(((gt_img-rd)**2).mean(dim=(1,2,3)))), rd

print("=== scale sweep (view0 GS -> view0 pose) ===")
best_f, best_p = None, -1
for f in np.geomspace(0.01, 2.0, 45):
    e = extr_t[:1].clone(); e[:, :3, 3] *= f
    p, _ = render_psnr(g0, e, intr_n[:1], gt[:1], 1)
    p = p.item()
    print(f"  factor {f:.4f}: {p:.2f} dB")
    if p > best_p: best_p, best_f = p, f

print(f"\nBEST factor = {best_f:.4f}  ({best_p:.2f} dB)")

os.makedirs("/workspace/results/sweep", exist_ok=True)
e = extr_t[:1].clone(); e[:, :3, 3] *= best_f
_, rd = render_psnr(g0, e, intr_n[:1], gt[:1], 1)
imageio.imwrite("/workspace/results/sweep/best_single.png", (rd[0].cpu().numpy()*255).astype(np.uint8))

e = extr_t.clone(); e[:, :3, 3] *= best_f
p, rd = render_psnr(g, e, intr_n, gt, 12)
print(f"\n=== all 12 views @ best factor: MEAN {p.mean().item():.2f} dB ===")
for i, v in enumerate(p): print(f"  view {idx[i]:3d}: {v.item():.2f} dB")
imageio.imwrite("/workspace/results/sweep/best_all.png", (rd[0].cpu().numpy()*255).astype(np.uint8))
print("Saved to /workspace/results/sweep/")