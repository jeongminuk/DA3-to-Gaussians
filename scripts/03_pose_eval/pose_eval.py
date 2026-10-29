"""
pose_eval.py
Camera-pose evaluation for Experiment #1 (IVL-SYNTHSFM-v2 / Statue).
COLMAP vs DA3, aligned to GT with Umeyama (Kabsch + scale).

Convention used throughout:
  - Poses are camera-to-world (c2w) 4x4 matrices.
  - Camera center C = t of the c2w matrix (translation column).
  - Camera-to-world rotation R = 3x3 rotation block of c2w.
If your source gives world-to-camera (COLMAP images.txt: qvec/tvec are w2c),
convert with w2c_to_c2w() before passing in.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pose conversions
# ---------------------------------------------------------------------------
def w2c_to_c2w(R_w2c, t_w2c):
    """COLMAP stores world->camera (X_cam = R X_world + t).
    Returns camera-to-world (R_c2w, center C)."""
    R_c2w = R_w2c.T
    C = -R_w2c.T @ t_w2c
    return R_c2w, C


# OpenGL (Blender) <-> OpenCV (COLMAP) camera-frame flip: flip Y and Z axes.
GL2CV = np.diag([1.0, -1.0, -1.0])


def convert_cam_frame(R_c2w, flip=GL2CV):
    """Re-express a c2w rotation in the other camera-axis convention.
    Apply to ONE side (est or gt) so both share the same local frame."""
    return R_c2w @ flip


# ---------------------------------------------------------------------------
# Umeyama alignment (Umeyama 1991) -- similarity transform src -> dst
# ---------------------------------------------------------------------------
def umeyama(src, dst, with_scale=True):
    """
    src, dst : (N,3) corresponding points.
    Returns s (float), R (3,3), t (3,) s.t.  dst ~= s * R @ src + t.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n, dim = src.shape

    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # cross-covariance (dst, src)
    Sigma = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1  # avoid reflection

    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    s = (np.trace(np.diag(D) @ S) / var_src) if with_scale else 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rotation_geodesic_deg(R_a, R_b):
    """Angle (deg) of relative rotation between two 3x3 rotations."""
    Rrel = R_a.T @ R_b
    c = (np.trace(Rrel) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def evaluate_poses(centers_est, R_est, centers_gt, R_gt,
                   with_scale=True, scene_extent=None):
    """
    centers_est/gt : (N,3) camera centers (world).
    R_est/gt       : (N,3,3) camera-to-world rotations, SAME local convention.
    Returns dict of metrics + per-view arrays.
    """
    centers_est = np.asarray(centers_est, float)
    centers_gt = np.asarray(centers_gt, float)
    R_est = np.asarray(R_est, float)
    R_gt = np.asarray(R_gt, float)

    # Fit similarity transform on centers (est -> gt).
    s, R_align, t_align = umeyama(centers_est, centers_gt, with_scale)

    centers_aln = (s * (R_align @ centers_est.T)).T + t_align
    trans_err = np.linalg.norm(centers_aln - centers_gt, axis=1)  # (N,)

    # Apply the SAME global rotation to est c2w rotations before comparing.
    rot_err = np.array([
        rotation_geodesic_deg(R_align @ R_est[i], R_gt[i])
        for i in range(len(R_est))
    ])

    def rmse(x):
        return float(np.sqrt((x ** 2).mean()))

    out = {
        "scale": float(s),
        "n": int(len(R_est)),
        "trans_rmse": rmse(trans_err),
        "trans_mean": float(trans_err.mean()),
        "trans_median": float(np.median(trans_err)),
        "rot_mean_deg": float(rot_err.mean()),
        "rot_median_deg": float(np.median(rot_err)),
        "trans_err": trans_err,
        "rot_err": rot_err,
    }
    if scene_extent:
        out["trans_rmse_pct"] = 100.0 * out["trans_rmse"] / scene_extent
    return out


# ---------------------------------------------------------------------------
# GT / prediction loaders  -- STUBS: fill in once the pose file format is known
# ---------------------------------------------------------------------------
def load_gt_poses(cameras_csv, to_opencv=True):
    """
    Parse IVL-SYNTHSFM-v2 cameras.csv.
    Columns: image_number, position_{x,y,z}, rotation_{w,x,y,z} (quat),
             lookat_{x,y,z}, depth_of_field, motion_blur,
             sun_azimuth, sun_inclination.

    position_* = camera CENTER (c2w translation), world coords.
    rotation_* = camera orientation quaternion, OpenGL frame (cam looks -Z).

    Returns dict with:
      names   : (N,) image ids as strings
      centers : (N,3) camera centers
      R_c2w   : (N,3,3) cam-to-world rotations. If to_opencv, expressed in
                OpenCV frame (cam looks +Z) so they match COLMAP/DA3.
      meta    : dict of dof/mb/sun (constant per scene)
    """
    import csv
    from scipy.spatial.transform import Rotation

    names, C, R = [], [], []
    meta = {}
    with open(cameras_csv, newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["image_number"])
            C.append([float(row["position_x"]),
                      float(row["position_y"]),
                      float(row["position_z"])])
            # scipy expects xyzw; file is wxyz
            q = [float(row["rotation_x"]), float(row["rotation_y"]),
                 float(row["rotation_z"]), float(row["rotation_w"])]
            Rgl = Rotation.from_quat(q).as_matrix()
            R.append(Rgl @ GL2CV if to_opencv else Rgl)
        meta = {"depth_of_field": row["depth_of_field"],
                "motion_blur": row["motion_blur"],
                "sun_azimuth": row["sun_azimuth"],
                "sun_inclination": row["sun_inclination"]}

    return {"names": names,
            "centers": np.array(C),
            "R_c2w": np.array(R),
            "convention": "OpenCV" if to_opencv else "OpenGL",
            "meta": meta}


def load_colmap_poses(images_txt):
    """
    Parse COLMAP images.txt.  Each image = 2 lines; the pose line has 10 fields:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    QW..QZ / TX..TZ are WORLD->CAMERA (OpenCV frame, cam looks +Z).
    Returns names, centers, R_c2w (OpenCV) -- matches GT loaded with to_opencv=True.
    """
    from scipy.spatial.transform import Rotation

    names, C, R = [], [], []
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.split()
            if len(t) != 10:          # skip the POINTS2D line (len is a multiple of 3)
                continue
            qw, qx, qy, qz = map(float, t[1:5])
            tx, ty, tz = map(float, t[5:8])
            name = t[9]
            R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            R_c2w, center = w2c_to_c2w(R_w2c, np.array([tx, ty, tz]))
            names.append(name)
            C.append(center)
            R.append(R_c2w)
    return {"names": names, "centers": np.array(C),
            "R_c2w": np.array(R), "convention": "OpenCV"}


def load_colmap_recon(colmap_dir):
    """
    Load a COLMAP reconstruction via pycolmap the same way gsplat's Parser does:
    read cam_from_world (w2c), invert to c2w, sort by image name.
    colmap_dir = .../sparse/0  (binary or txt model).
    Returns names, centers, R_c2w (OpenCV) -- matches GT to_opencv=True.
    """
    import pycolmap
    rec = pycolmap.Reconstruction(colmap_dir)
    names, C, R = [], [], []
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        w2c = np.eye(4)
        w2c[:3, :4] = np.asarray(cfw.matrix(), dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        names.append(im.name)
        C.append(c2w[:3, 3])
        R.append(c2w[:3, :3])
    order = np.argsort(names)                       # gsplat sorts by name
    names = [names[i] for i in order]
    return {"names": names,
            "centers": np.array(C)[order],
            "R_c2w": np.array(R)[order],
            "convention": "OpenCV"}


def load_da3_poses(npy_path, names_txt=None, source="camera_decoder",
                   to_opencv_flip=None):
    """
    Load DA3 predicted extrinsics saved as a .npy of shape (N,4,4) or (N,3,4).

    DA3 stores poses differently per path (see da3.py):
      - camera_decoder (default): output.extrinsics = affine_inverse(c2w) => it is W2C
      - ray (use_ray_pose=True) : output.extrinsics is already C2W
    Pass source accordingly so we return C2W consistently.

    names_txt: optional text file, one image id per line, SAME order as npy rows.
    to_opencv_flip: if DA3's camera frame differs from GT (rotation error ~180
               deg after alignment), pass GL2CV. Leave None first, check, then set.
    """
    import os
    ext = np.asarray(np.load(npy_path), float)
    if ext.ndim != 3 or ext.shape[-1] != 4:
        raise ValueError(f"expected (N,4,4)/(N,3,4), got {ext.shape}")

    C, R = [], []
    for M in ext:
        if M.shape[0] == 3:                      # (3,4) -> (4,4)
            M = np.vstack([M, [0, 0, 0, 1.0]])
        if source == "camera_decoder":           # stored as W2C -> invert
            Rc2w, center = w2c_to_c2w(M[:3, :3], M[:3, 3])
        else:                                    # ray path: already C2W
            Rc2w, center = M[:3, :3], M[:3, 3]
        if to_opencv_flip is not None:
            Rc2w = Rc2w @ to_opencv_flip
        R.append(Rc2w); C.append(center)

    if names_txt and os.path.exists(names_txt):
        names = [l.strip() for l in open(names_txt) if l.strip()]
    else:
        names = [str(i) for i in range(len(ext))]

    return {"names": names, "centers": np.array(C),
            "R_c2w": np.array(R), "convention": "OpenCV", "source": source}


def _stem(name):
    """'0007.jpg' -> '0007' so COLMAP/DA3/GT ids match regardless of extension."""
    import os
    return os.path.splitext(os.path.basename(name))[0]


def match_and_evaluate(est, gt, with_scale=True, scene_extent=None):
    """
    est, gt : dicts from the loaders (need 'names','centers','R_c2w').
    Matches by image-id stem (handles subset registration / extensions),
    then aligns est->gt (Umeyama on matched centers) and computes errors.
    """
    gt_idx = {_stem(n): i for i, n in enumerate(gt["names"])}
    e_i, g_i, matched = [], [], []
    for j, n in enumerate(est["names"]):
        s = _stem(n)
        if s in gt_idx:
            e_i.append(j); g_i.append(gt_idx[s]); matched.append(s)

    if len(matched) < 3:
        raise RuntimeError(f"Only {len(matched)} matched views; need >=3 to align.")

    res = evaluate_poses(
        est["centers"][e_i], est["R_c2w"][e_i],
        gt["centers"][g_i], gt["R_c2w"][g_i],
        with_scale=with_scale, scene_extent=scene_extent)
    res["n_matched"] = len(matched)
    res["n_est"] = len(est["names"])
    res["n_gt"] = len(gt["names"])
    return res


if __name__ == "__main__":
    print("pose_eval.py — import and call evaluate_poses(). Run self-test via test_pose_eval.py")