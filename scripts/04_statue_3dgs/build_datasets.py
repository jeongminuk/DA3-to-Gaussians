"""
build_datasets.py   (gsplat container)
Statue few-shot: build one COLMAP-format dataset per (pose source, N) for gsplat.

  colmap       K, poses, points from COLMAP (exp/Statue/fs/colmap_<N>)
  da3_decoder  K, poses, depth->points from DA3 camera decoder
  da3_ray      K, poses, depth->points from DA3 ray head

Each dataset holds the N train views plus every test view. Test-view poses are
GT poses mapped into that method's world frame with a Sim(3) fitted on the train
camera centers (GT -> method), so all three methods are evaluated on the same
held-out images. DA3 init points are sampled uniformly (no conf filter) from the
unprojected depth, with the same count as COLMAP's sparse points for that N.

Output: /workspace/statue3dgs/data/<method>_<N>/{images/, sparse/0/, split.json, pose_eval.json}
"""
import json, os
import numpy as np
from read_write_model import read_model, write_model, Camera, Image, Point3D
from pose_eval import load_gt_poses, umeyama, rotation_geodesic_deg

FS = "/workspace/data/Statue/fs"
COLMAP = "/workspace/exp/Statue/fs/colmap_{N}/sparse/0"
DA3 = "/workspace/statue3dgs/da3/{N}_{mode}.npz"
OUT = "/workspace/statue3dgs/data"
W, H = 1920, 1080
rng = np.random.default_rng(0)


def rotmat2qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def load_colmap(N):
    cams, imgs, pts = read_model(COLMAP.format(N=N))
    assert len(cams) == 1
    K = list(cams.values())[0].params                       # fx fy cx cy (PINHOLE)
    c2w = {}
    for im in imgs.values():
        w2c = np.eye(4); w2c[:3, :3] = im.qvec2rotmat(); w2c[:3, 3] = im.tvec
        c2w[os.path.splitext(im.name)[0]] = np.linalg.inv(w2c)
    xyz = np.array([p.xyz for p in pts.values()])
    rgb = np.array([p.rgb for p in pts.values()])
    return K, c2w, xyz, rgb


def load_da3(N, mode, n_points):
    d = np.load(DA3.format(N=N, mode=mode))
    ext, Ks, depth, images = d["extrinsics"], d["intrinsics"], d["depth"], d["images"]
    V, h, w = depth.shape
    sx, sy = W / w, H / h
    K = np.array([Ks[:, 0, 0].mean() * sx, Ks[:, 1, 1].mean() * sy,
                  Ks[:, 0, 2].mean() * sx, Ks[:, 1, 2].mean() * sy])
    c2w, xyz, rgb = {}, [], []
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    for i, name in enumerate(d["names"]):
        w2c = np.eye(4); w2c[:3] = ext[i]
        M = np.linalg.inv(w2c)
        c2w[os.path.splitext(str(name))[0]] = M
        cam = (pix @ np.linalg.inv(Ks[i]).T) * depth[i].reshape(-1, 1)
        xyz.append(cam @ M[:3, :3].T + M[:3, 3])
        rgb.append(images[i].reshape(-1, 3))
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    keep = rng.choice(len(xyz), n_points, replace=False)
    return K, c2w, xyz[keep], rgb[keep], float(d["seconds"])


def main():
    gt = load_gt_poses(f"{FS}/cameras.csv")                  # OpenCV c2w
    gt_c2w = {}
    for n, C, R in zip(gt["names"], gt["centers"], gt["R_c2w"]):
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = C; gt_c2w[n] = M
    extent = np.linalg.norm(gt["centers"] - gt["centers"].mean(0), axis=1).max()
    splits = json.load(open(f"{FS}/splits.json"))

    for N in (12, 24, 50):
        train, test = splits[str(N)]["train"], splits[str(N)]["test"]
        _, _, cxyz, _ = load_colmap(N)
        for method in ("colmap", "da3_decoder", "da3_ray"):
            if method == "colmap":
                K, c2w, xyz, rgb = load_colmap(N); secs = None
            else:
                K, c2w, xyz, rgb, secs = load_da3(N, method.split("_")[1], len(cxyz))
            assert sorted(c2w) == sorted(train), (method, N)

            # Sim(3) GT -> method frame, fitted on train centers
            src = np.array([gt_c2w[n][:3, 3] for n in train])
            dst = np.array([c2w[n][:3, 3] for n in train])
            s, R, t = umeyama(src, dst)
            # pose accuracy of the train views (method -> GT direction, as in compare_poses)
            s2, R2, t2 = umeyama(dst, src)
            terr = np.linalg.norm((s2 * (R2 @ dst.T)).T + t2 - src, axis=1)
            rerr = [rotation_geodesic_deg(R2 @ c2w[n][:3, :3], gt_c2w[n][:3, :3]) for n in train]
            for n in test:
                M = np.eye(4); M[:3, :3] = R @ gt_c2w[n][:3, :3]; M[:3, 3] = s * R @ gt_c2w[n][:3, 3] + t
                c2w[n] = M

            out = f"{OUT}/{method}_{N}"
            os.makedirs(f"{out}/sparse/0", exist_ok=True)
            os.makedirs(f"{out}/images", exist_ok=True)
            cameras = {1: Camera(id=1, model="PINHOLE", width=W, height=H, params=np.asarray(K, float))}
            images = {}
            for i, n in enumerate(sorted(c2w), start=1):
                w2c = np.linalg.inv(c2w[n])
                images[i] = Image(id=i, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3],
                                  camera_id=1, name=f"{n}.jpg",
                                  xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, dtype=int))
                link = f"{out}/images/{n}.jpg"
                if not os.path.lexists(link):
                    os.symlink(f"../../../../data/Statue/fs/{n}.jpg", link)
            points = {i: Point3D(id=i, xyz=p, rgb=np.clip(c, 0, 255).astype(np.uint8), error=0.0,
                                 image_ids=np.zeros(0, dtype=int), point2D_idxs=np.zeros(0, dtype=int))
                      for i, (p, c) in enumerate(zip(xyz, rgb), start=1)}
            write_model(cameras, images, points, f"{out}/sparse/0", ".bin")
            json.dump({"train": [f"{n}.jpg" for n in train], "test": [f"{n}.jpg" for n in test]},
                      open(f"{out}/split.json", "w"))
            pe = {"method": method, "N": N, "n_points": len(xyz), "K": list(map(float, K)),
                  "trans_rmse_pct": float(100 * np.sqrt((terr ** 2).mean()) / extent),
                  "rot_mean_deg": float(np.mean(rerr)), "rot_median_deg": float(np.median(rerr)),
                  "da3_seconds": secs}
            json.dump(pe, open(f"{out}/pose_eval.json", "w"), indent=1)
            print(json.dumps(pe))


if __name__ == "__main__":
    main()
