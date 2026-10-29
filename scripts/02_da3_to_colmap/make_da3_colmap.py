import os, shutil, numpy as np
from depth_anything_3.utils.read_write_model import (
    read_model, write_model, Image as ColmapImage
)

SRC = "/workspace/data/tandt/truck/sparse/0"
DST = "/workspace/data/truck_da3pose/sparse/0"
NPZ = "/workspace/results/da3_poses_all.npz"
os.makedirs(DST, exist_ok=True)

d = np.load(NPZ, allow_pickle=True)
da3_e, col_e, names = d["extrinsics"], d["colmap_extrinsics"], d["names"]

def to4(e):
    e = np.asarray(e, dtype=np.float64)
    if e.shape[-2] == 3:
        e = np.concatenate([e, np.tile(np.array([[[0,0,0,1.]]]), (e.shape[0],1,1))], axis=1)
    return e
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

# DA3 -> COLMAP 좌표계 (방향 주의: 이번엔 DA3를 COLMAP에 맞춤)
sc, R_a, t_a = umeyama(cen(da3_4), cen(col_4))
pred_c = sc*(R_a @ cen(da3_4).T).T + t_a
resid = np.linalg.norm(pred_c - cen(col_4), axis=1)
scene = np.linalg.norm(cen(col_4).max(0) - cen(col_4).min(0))
print(f"Sim3 scale {sc:.5f}")
print(f"정렬 오차: mean {resid.mean():.4f}, max {resid.max():.4f} "
      f"(씬 {scene:.3f} 대비 {100*resid.mean()/scene:.2f}%)")

# w2c 변환: R_new = R_da3 @ R_a^T,  t_new = sc*t_da3 - R_new @ t_a
R_new = da3_4[:,:3,:3] @ R_a.T
t_new = sc*da3_4[:,:3,3] - np.einsum("nij,j->ni", R_new, t_a)

def mat2quat(R):  # -> (w,x,y,z), COLMAP qvec 규약
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[0,1]+R[1,0], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[0,2]+R[2,0], R[1,2]+R[2,1], R[2,2]-R[0,0]-R[1,1], 0],
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]],
    ])/3.0
    w, V = np.linalg.eigh(K)
    q = V[[3,0,1,2], np.argmax(w)]
    return q if q[0] >= 0 else -q

cameras, images, points3D = read_model(SRC)
name2id = {v.name: k for k, v in images.items()}
new_images = {}
for i, nm in enumerate(names):
    nm = str(nm)
    iid = name2id[nm]
    old = images[iid]
    new_images[iid] = ColmapImage(
        id=old.id, qvec=mat2quat(R_new[i]), tvec=t_new[i],
        camera_id=old.camera_id, name=old.name,
        xys=old.xys, point3D_ids=old.point3D_ids,
    )

write_model(cameras, new_images, points3D, DST, ext=".bin")
print(f"\nwrote -> {DST}")
for f in sorted(os.listdir(DST)):
    print(f"  {f}  ({os.path.getsize(os.path.join(DST,f))/1e6:.1f} MB)")

# images 폴더 심볼릭 링크(용량 절약)
img_link = "/workspace/data/truck_da3pose/images"
if not os.path.exists(img_link):
    os.symlink("/workspace/data/tandt/truck/images", img_link)
print(f"images -> {img_link}")