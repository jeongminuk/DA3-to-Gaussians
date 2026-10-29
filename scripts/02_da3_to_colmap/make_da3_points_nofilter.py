import os, sys, shutil, numpy as np, torch
from depth_anything_3.utils.geometry import get_world_rays, sample_image_grid, affine_inverse
from depth_anything_3.utils.read_write_model import read_model, write_model, Point3D

N_POINTS = int(sys.argv[1]) if len(sys.argv) > 1 else 136029   # 인자로 개수 지정
SRC = "/workspace/data/truck_da3pose/sparse/0"                  # 기존 DA3 포즈
DST = f"/workspace/data/truck_da3nofilter_{N_POINTS//1000}k/sparse/0"
os.makedirs(DST, exist_ok=True)

d = np.load("/workspace/results/da3_geometry.npz", allow_pickle=True)
depth, conf, imgs = d["depth"], d["conf"], d["images"]
da3_e, col_e = d["extrinsics"], d["colmap_extrinsics"]
V, H, W = depth.shape
print(f"{V}장 {H}x{W} | 전체 픽셀 {V*H*W:,} | 목표 {N_POINTS:,}")

def to4(e):
    e = np.asarray(e, dtype=np.float64)
    return e if e.shape[-2] == 4 else np.concatenate(
        [e, np.tile(np.array([[[0,0,0,1.]]]), (e.shape[0],1,1))], axis=1)

def cen(E):
    R, t = E[:,:3,:3], E[:,:3,3]
    return np.einsum("nij,nj->ni", R.transpose(0,2,1), -t)

def umeyama(s, dst_):
    ms, md = s.mean(0), dst_.mean(0); S, D = s-ms, dst_-md
    C = D.T @ S / len(s); U, sig, Vt = np.linalg.svd(C); F = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: F[2,2] = -1
    R = U @ F @ Vt; sc = (sig*np.diag(F)).sum()/(S**2).sum()*len(s)
    return sc, R, md - sc*R@ms

da3_4, col_4 = to4(da3_e), to4(col_e)
sc, R_a, t_a = umeyama(cen(da3_4), cen(col_4))
print(f"Sim3 scale {sc:.5f}")

# 뷰별 unprojection (gs_adapter와 동일 방식)
dev = "cuda"
xy_ray, _ = sample_image_grid((H, W), dev)
c2w_all = affine_inverse(torch.from_numpy(da3_4).float().to(dev))
intr_n = torch.from_numpy(np.asarray(d["intrinsics"])).float().to(dev).clone()
intr_n[:,0,:] /= W; intr_n[:,1,:] /= H

pts_list = []
with torch.no_grad():
    for i in range(V):
        c2w_i = c2w_all[i][None,None].expand(H, W, 4, 4)
        K_i   = intr_n[i][None,None].expand(H, W, 3, 3)
        org, dirs = get_world_rays(xy_ray, c2w_i, K_i)
        dep = torch.from_numpy(depth[i]).float().to(dev)[..., None]
        pts_list.append((org + dirs*dep).reshape(-1, 3).cpu().numpy())
pts_da3 = np.concatenate(pts_list).astype(np.float64)
cols = imgs.reshape(-1, 3)
cf = conf.reshape(-1)
print(f"unprojected {len(pts_da3):,}")

# 뷰별 균등 샘플링 + confidence 필터 (뷰 내 상위 50% 후보 중 무작위)
if N_POINTS < len(pts_da3):
    conf_v = conf.reshape(V, -1)
    rng = np.random.default_rng(42)
    base = N_POINTS // V
    extra = N_POINTS - base * V
    sel_list = []
    for i in range(V):
        quota = base + (1 if i < extra else 0)
        c = conf_v[i]
        cand = np.arange(len(c))
        if len(cand) > quota:
            cand = rng.choice(cand, quota, replace=False)
        sel_list.append(cand + i * H * W)
    sel = np.sort(np.concatenate(sel_list))
    pts_da3, cols, cf = pts_da3[sel], cols[sel], cf[sel]
print(f"selected {len(pts_da3):,} (뷰당 ~{N_POINTS//V:,}) | conf [{cf.min():.3f}, {cf.max():.3f}]")

# Sim(3) 적용: DA3 -> COLMAP 좌표계
pts = sc*(R_a @ pts_da3.T).T + t_a

# sanity: 백분위수로 비교 (outlier 영향 배제)
cameras, images, points3D_old = read_model(SRC)
old_xyz = np.array([p.xyz for p in points3D_old.values()])
print(f"\n[sanity] COLMAP 1~99%: {np.percentile(old_xyz,1,axis=0).round(2)} ~ {np.percentile(old_xyz,99,axis=0).round(2)}")
print(f"[sanity] DA3    1~99%: {np.percentile(pts,1,axis=0).round(2)} ~ {np.percentile(pts,99,axis=0).round(2)}")
print(f"[sanity] 카메라 궤적 크기(COLMAP): {np.linalg.norm(cen(col_4).max(0)-cen(col_4).min(0)):.2f}")

points3D = {i+1: Point3D(id=i+1, xyz=pts[i], rgb=cols[i].astype(np.uint8),
                         error=np.array(0.0),
                         image_ids=np.array([], dtype=np.int64),
                         point2D_idxs=np.array([], dtype=np.int64))
            for i in range(len(pts))}
write_model(cameras, images, points3D, DST, ext=".bin")

img_dst = os.path.dirname(os.path.dirname(DST)) + "/images"
if not os.path.exists(img_dst):
    shutil.copytree("/workspace/data/tandt/truck/images", img_dst)
print(f"\nwrote -> {os.path.dirname(os.path.dirname(DST))}")
for f in sorted(os.listdir(DST)):
    print(f"  {f}  ({os.path.getsize(os.path.join(DST,f))/1e6:.1f} MB)")