"""
eval_tto.py   (gsplat container, run from gsplat/examples)
Evaluate a trained Statue run on its held-out test views, twice:
  aligned : test poses = GT mapped by the train-view Sim(3) (same as gsplat's own val)
  tto     : + test-time pose optimization; Gaussians frozen, only each test camera's
            6-DoF pose is refined against the test image (Adam, L1), as in pose-free
            NVS evaluation (e.g. InstantSplat). Applied identically to every method.
Writes <result_dir>/stats/eval_tto.json and a few side-by-side JPEGs to <result_dir>/compare/.
Usage: python /workspace/statue3dgs/eval_tto.py <run_name> [iters]
"""
import json, os, sys
import imageio.v2 as imageio
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from datasets.colmap import Parser
from gsplat.rendering import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

RUN = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
DATA = f"/workspace/statue3dgs/data/{RUN}"
RES = f"/workspace/statue3dgs/results/{RUN}"
SAVE_VIEWS = {"0006.jpg", "0026.jpg", "0051.jpg", "0063.jpg"}   # test views common to N=12/24/50
dev = "cuda"

parser = Parser(DATA, factor=1, normalize=True, test_every=8)      # same as simple_trainer
test = json.load(open(f"{DATA}/split.json"))["test"]
splats = torch.load(f"{RES}/ckpts/ckpt_29999_rank0.pt", map_location=dev, weights_only=False)["splats"]
means, quats = splats["means"], F.normalize(splats["quats"], dim=-1)
scales, opac = torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"])
colors = torch.cat([splats["sh0"], splats["shN"]], 1)

psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)


def render(c2w, K, W, H):
    out, _, _ = rasterization(means, quats, scales, opac, colors, torch.linalg.inv(c2w)[None], K[None],
                              W, H, sh_degree=3, near_plane=0.01, far_plane=1e10, packed=False,
                              rasterize_mode="classic")
    return out[0].clamp(0, 1)


def so3_exp(w):
    th = w.norm() + 1e-12
    k = w / th
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def metrics(img, gt):
    a, b = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
    return [psnr(a, b).item(), ssim(a, b).item(), lpips(a, b).item()]


os.makedirs(f"{RES}/compare", exist_ok=True)
res = {"aligned": [], "tto": []}
for name in test:
    i = parser.image_names.index(name)
    c2w0 = torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device=dev)
    K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], dtype=torch.float32, device=dev)
    gt = torch.tensor(imageio.imread(parser.image_paths[i])[..., :3], device=dev).float() / 255
    H, W = gt.shape[:2]
    with torch.no_grad():
        img0 = render(c2w0, K, W, H)
    res["aligned"].append(metrics(img0, gt))

    w = torch.zeros(3, device=dev, requires_grad=True)
    t = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, t], lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.01 ** (1 / ITERS))
    gt_s = F.interpolate(gt.permute(2, 0, 1)[None], scale_factor=0.5, mode="area")
    Ks = K.clone(); Ks[:2] *= 0.5
    for _ in range(ITERS):                                  # optimize at half resolution
        T = torch.eye(4, device=dev); T[:3, :3] = so3_exp(w); T[:3, 3] = t
        img = render(c2w0 @ T, Ks, W // 2, H // 2)
        loss = (img.permute(2, 0, 1)[None] - gt_s).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        T = torch.eye(4, device=dev); T[:3, :3] = so3_exp(w); T[:3, 3] = t
        img1 = render(c2w0 @ T, K, W, H)
    res["tto"].append(metrics(img1, gt))

    if name in SAVE_VIEWS:
        for tag, im in (("aligned", img0), ("tto", img1)):
            imageio.imwrite(f"{RES}/compare/{name[:-4]}_{tag}.jpg", (im.cpu().numpy() * 255).astype(np.uint8), quality=90)

out = {k: dict(zip(["psnr", "ssim", "lpips"], np.mean(v, 0).tolist())) for k, v in res.items()}
out.update(run=RUN, n_test=len(test), tto_iters=ITERS, num_GS=int(len(means)),
           per_view={k: {n: v for n, v in zip(test, res[k])} for k in res})
json.dump(out, open(f"{RES}/stats/eval_tto.json", "w"), indent=1)
print(RUN, {k: out[k] for k in ("aligned", "tto")})
