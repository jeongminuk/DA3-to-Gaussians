"""
find_flip.py  — brute-force the camera-frame transform that best aligns
DA3 poses to GT. Tries all 48 signed axis-permutation matrices (det=+1 only,
i.e. proper rotations) as a right-multiply on c2w rotations, and reports the
one giving the lowest mean rotation error after Umeyama alignment.

Usage (host):
  python find_flip.py \
    --cameras_csv /home/minuk0221/gsplat_workspace/data/Statue/fs/cameras.csv \
    --npy  /home/minuk0221/da3_workspace/exp/Statue/fs/da3_12_ray/pred_extrinsics.npy \
    --names /home/minuk0221/da3_workspace/exp/Statue/fs/da3_12_ray/names.txt \
    --source ray
"""
import argparse, itertools
import numpy as np
from pose_eval import load_gt_poses, load_da3_poses, match_and_evaluate


def all_proper_axis_transforms():
    """48 signed permutations; keep the 24 with det=+1 (proper rotations)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.eye(3)[list(perm)]
        for signs in itertools.product([1, -1], repeat=3):
            M = P * np.array(signs)[None, :]
            if abs(np.linalg.det(M) - 1) < 1e-6:
                mats.append(M)
    return mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras_csv", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--source", default="ray",
                    choices=["ray", "camera_decoder"])
    args = ap.parse_args()

    gt = load_gt_poses(args.cameras_csv)
    ext = np.linalg.norm(gt["centers"] - gt["centers"].mean(0), axis=1).max()

    results = []
    for M in all_proper_axis_transforms():
        est = load_da3_poses(args.npy, args.names, source=args.source,
                             to_opencv_flip=M)
        r = match_and_evaluate(est, gt, scene_extent=ext)
        results.append((r["rot_mean_deg"], r["trans_rmse_pct"], M))
    results.sort(key=lambda x: x[0])

    print("best 3 camera-frame transforms (lowest rotation error):")
    for rot, tpct, M in results[:3]:
        print(f"  rot_mean={rot:7.3f}°  trans={tpct:6.2f}%   M(diag/rows)=\n{M}")
    print("\nworst (for contrast):")
    rot, tpct, M = results[-1]
    print(f"  rot_mean={rot:7.3f}°  trans={tpct:6.2f}%")


if __name__ == "__main__":
    main()