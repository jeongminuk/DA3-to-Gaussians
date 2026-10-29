import os, glob, numpy as np, torch, imageio
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.gs_renderer import render_3dgs

MODEL_DIR = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
EX = "/workspace/Depth-Anything-3/assets/examples/SOH"

files = sorted(glob.glob(os.path.join(EX, "*")))
files = [f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))]
print(f"example images: {len(files)}")

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
# 포즈 없이 (모델이 알아서 추정) — 공식 기본 사용법
pred = model.inference(image=files, infer_gs=True)

g = pred.gaussians
H, W = pred.processed_images.shape[1:3]
print(f"H={H} W={W}, N={g.means.shape[1]}")

# Gaussian 통계
print("\n=== Gaussian stats ===")
m = g.means[0]
print(f"means  range: [{m.min().item():.4f}, {m.max().item():.4f}]  extent={(m.max()-m.min()).item():.4f}")
s = g.scales[0]
print(f"scales min/mean/median/max: {s.min().item():.6f} / {s.mean().item():.6f} / {s.median().item():.6f} / {s.max().item():.6f}")
o = g.opacities[0]
print(f"opacity mean={o.mean().item():.4f}  >0.5 ratio={(o>0.5).float().mean().item():.3f}")
print(f"depth range: [{pred.depth.min():.4f}, {pred.depth.max():.4f}]")

extr = np.asarray(pred.extrinsics)
if extr.shape[-2] == 3:
    extr = np.concatenate([extr, np.tile(np.array([[[0,0,0,1.]]]), (extr.shape[0],1,1))], axis=1)
e = torch.from_numpy(extr).float().cuda()
i_t = torch.from_numpy(np.asarray(pred.intrinsics)).float().cuda()
i_n = i_t.clone(); i_n[...,0,:] /= W; i_n[...,1,:] /= H

with torch.no_grad():
    c, _ = render_3dgs(extrinsics=e, intrinsics=i_n, image_shape=(H,W),
                       gaussian=g, num_view=e.shape[0])
gt = torch.from_numpy(pred.processed_images).float().cuda()/255.0
rd = c.permute(0,2,3,1).clamp(0,1)
p = -10*torch.log10(((gt-rd)**2).mean(dim=(1,2,3)))
print(f"\n=== official example, context-view PSNR: MEAN {p.mean().item():.2f} dB ===")

os.makedirs("/workspace/results/control", exist_ok=True)
for i in range(min(2, rd.shape[0])):
    imageio.imwrite(f"/workspace/results/control/render_{i}.png", (rd[i].cpu().numpy()*255).astype(np.uint8))
    imageio.imwrite(f"/workspace/results/control/gt_{i}.png", pred.processed_images[i])
print("Saved to /workspace/results/control/")