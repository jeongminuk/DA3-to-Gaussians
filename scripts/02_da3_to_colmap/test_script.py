import numpy as np, torch
from depth_anything_3.utils.geometry import get_world_rays, sample_image_grid, affine_inverse

d = np.load("/workspace/results/da3_geometry.npz", allow_pickle=True)
depth, da3_e, col_e = d["depth"], d["extrinsics"], d["colmap_extrinsics"]
V,H,W = depth.shape

def to4(e):
    e=np.asarray(e,dtype=np.float64)
    return e if e.shape[-2]==4 else np.concatenate([e,np.tile(np.array([[[0,0,0,1.]]]),(e.shape[0],1,1))],axis=1)
def cen(E):
    R,t=E[:,:3,:3],E[:,:3,3]; return np.einsum("nij,nj->ni",R.transpose(0,2,1),-t)

da3_4, col_4 = to4(da3_e), to4(col_e)
print("=== 카메라 중심 범위 ===")
print("DA3   :", cen(da3_4).min(0).round(3), "~", cen(da3_4).max(0).round(3))
print("COLMAP:", cen(col_4).min(0).round(3), "~", cen(col_4).max(0).round(3))
print(f"DA3 카메라 궤적 크기: {np.linalg.norm(cen(da3_4).max(0)-cen(da3_4).min(0)):.3f}")
print(f"COLMAP 카메라 궤적 크기: {np.linalg.norm(cen(col_4).max(0)-cen(col_4).min(0)):.3f}")
print(f"→ 비율: {np.linalg.norm(cen(col_4).max(0)-cen(col_4).min(0))/np.linalg.norm(cen(da3_4).max(0)-cen(da3_4).min(0)):.3f}")

print(f"\n=== depth ===")
print(f"범위 [{depth.min():.3f}, {depth.max():.3f}], 중앙값 {np.median(depth):.3f}")

# 뷰 0의 unprojection을 DA3 좌표계에서 확인 (Sim3 적용 전)
dev="cuda"
xy,_ = sample_image_grid((H,W),dev)
c2w = affine_inverse(torch.from_numpy(da3_4[:1]).float().to(dev))
K = torch.from_numpy(np.asarray(d["intrinsics"])[:1]).float().to(dev).clone()
K[:,0,:]/=W; K[:,1,:]/=H
with torch.no_grad():
    org,dirs = get_world_rays(xy, c2w[0][None,None].expand(H,W,4,4), K[0][None,None].expand(H,W,3,3))
    print(f"\ndirs norm: min {dirs.norm(dim=-1).min():.4f}, max {dirs.norm(dim=-1).max():.4f}")
    print("  (1.0 근처면 단위벡터 → depth는 ray distance여야 함)")
    p = (org + dirs*torch.from_numpy(depth[0]).float().to(dev)[...,None]).reshape(-1,3).cpu().numpy()
print(f"\n뷰0 점군(DA3 좌표계) 범위: {p.min(0).round(3)} ~ {p.max(0).round(3)}")
print(f"뷰0 카메라 위치: {cen(da3_4[:1])[0].round(3)}")
print(f"카메라~점 거리: 중앙값 {np.median(np.linalg.norm(p-cen(da3_4[:1])[0],axis=1)):.3f}")
print(f"  (depth 중앙값 {np.median(depth[0]):.3f}과 비슷해야 정상)")