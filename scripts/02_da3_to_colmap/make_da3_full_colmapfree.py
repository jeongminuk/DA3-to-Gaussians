import os, sys, shutil, numpy as np, torch
from PIL import Image as PILImage
from depth_anything_3.utils.geometry import get_world_rays, sample_image_grid, affine_inverse
from depth_anything_3.utils.read_write_model import write_model, Camera, Image as ColmapImage, Point3D

N_POINTS = int(sys.argv[1]) if len(sys.argv) > 1 else 136029
IMG_SRC = "/workspace/data/tandt/truck/images"
DST = f"/workspace/data/truck_colmapfree_{N_POINTS//1000}k/sparse/0"
os.makedirs(DST, exist_ok=True)

d = np.load("/workspace/results/da3_geometry.npz", allow_pickle=True)
depth, conf, imgs = d["depth"], d["conf"], d["images"]
da3_e, da3_K, names = d["extrinsics"], d["intrinsics"], d["names"]
V, H, W = depth.shape

# 원본 해상도 확인
orig_W, orig_H = PILImage.open(os.path.join(IMG_SRC, str(names[0]))).size
sx, sy = orig_W / W, orig_H / H
print(f"처리 {W}x{H} → 원본 {orig_W}x{orig_H} (scale {sx:.3f}, {sy:.3f})")

def to4(e):
    e = np.asarray(e, dtype=np.float64)
    return e if e.shape[-2] == 4 else np.concatenate(
        [e, np.tile(np.array([[[0,0,0,1.]]]), (e.shape[0],1,1))], axis=1)
da3_4 = to4(da3_e)

# --- cameras.bin : DA3 예측 intrinsic (원본 해상도로 환산) ---
K0 = da3_K[0]
fx, fy = K0[0,0]*sx, K0[1,1]*sy
cx, cy = K0[0,2]*sx, K0[1,2]*sy
print(f"DA3 intrinsic → fx {fx:.2f}, fy {fy:.2f}, cx {cx:.2f}, cy {cy:.2f}")
cameras = {1: Camera(id=1, model="PINHOLE", width=orig_W, height=orig_H,
                     params=np.array([fx, fy, cx, cy]))}

# --- images.bin : DA3 포즈 (Sim(3) 없음, DA3 좌표계 그대로) ---
def mat2quat(R):
    K_ = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[0,1]+R[1,0], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[0,2]+R[2,0], R[1,2]+R[2,1], R[2,2]-R[0,0]-R[1,1], 0],
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]],
    ])/3.0
    w, Vv = np.linalg.eigh(K_)
    q = Vv[[3,0,1,2], np.argmax(w)]
    return q if q[0] >= 0 else -q

images = {}
for i in range(V):
    images[i+1] = ColmapImage(
        id=i+1, qvec=mat2quat(da3_4[i,:3,:3]), tvec=da3_4[i,:3,3],
        camera_id=1, name=str(names[i]),
        xys=np.zeros((0,2), dtype=np.float64),
        point3D_ids=np.zeros(0, dtype=np.int64))

# --- points3D.bin : depth unprojection (Sim(3) 없음) ---
dev = "cuda"
xy_ray, _ = sample_image_grid((H, W), dev)
c2w_all = affine_inverse(torch.from_numpy(da3_4).float().to(dev))
intr_n = torch.from_numpy(np.asarray(da3_K)).float().to(dev).clone()
intr_n[:,0,:] /= W; intr_n[:,1,:] /= H

pts_list = []
with torch.no_grad():
    for i in range(V):
        org, dirs = get_world_rays(
            xy_ray, c2w_all[i][None,None].expand(H,W,4,4),
            intr_n[i][None,None].expand(H,W,3,3))
        dep = torch.from_numpy(depth[i]).float().to(dev)[...,None]
        pts_list.append((org + dirs*dep).reshape(-1,3).cpu().numpy())
pts = np.concatenate(pts_list).astype(np.float64)
cols = imgs.reshape(-1,3)

# 뷰별 균등 무작위 (필터 없음)
rng = np.random.default_rng(42)
base, extra = N_POINTS // V, N_POINTS - (N_POINTS//V)*V
sel = np.sort(np.concatenate([
    rng.choice(H*W, base + (1 if i < extra else 0), replace=False) + i*H*W
    for i in range(V)]))
pts, cols = pts[sel], cols[sel]
print(f"points {len(pts):,} | 범위 1~99%: {np.percentile(pts,1,axis=0).round(2)} ~ {np.percentile(pts,99,axis=0).round(2)}")

points3D = {i+1: Point3D(id=i+1, xyz=pts[i], rgb=cols[i].astype(np.uint8),
                         error=np.array(0.0),
                         image_ids=np.array([], dtype=np.int64),
                         point2D_idxs=np.array([], dtype=np.int64))
            for i in range(len(pts))}

write_model(cameras, images, points3D, DST, ext=".bin")

img_dst = os.path.dirname(os.path.dirname(DST)) + "/images"
if not os.path.exists(img_dst):
    shutil.copytree(IMG_SRC, img_dst)
print(f"\nwrote -> {os.path.dirname(os.path.dirname(DST))}  (COLMAP 완전 미사용)")
for f in sorted(os.listdir(DST)):
    print(f"  {f}  ({os.path.getsize(os.path.join(DST,f))/1e6:.1f} MB)")