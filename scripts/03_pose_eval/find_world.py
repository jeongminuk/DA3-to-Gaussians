"""
find_world.py — find the WORLD-frame transform (incl. reflections) that maps
DA3 ray poses onto GT. Unlike find_flip, this transforms BOTH camera centers
and rotations by W (C'=W@C, R'=W@R) and tries all 48 signed axis permutations
(det = +1 AND -1), so world-frame reflections are covered.
"""
import argparse, itertools
import numpy as np
from pose_eval import load_gt_poses, load_da3_poses, evaluate_poses, _stem


def all_signed_perms():
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.eye(3)[list(perm)]
        for signs in itertools.product([1, -1], repeat=3):
            mats.append(P * np.array(signs)[None, :])
    return mats  # 48 total, det = +/-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras_csv", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--source", default="ray")
    args = ap.parse_args()

    gt = load_gt_poses(args.cameras_csv)
    extent = np.linalg.norm(gt["centers"] - gt["centers"].mean(0), axis=1).max()
    est = load_da3_poses(args.npy, args.names, source=args.source)

    # match by stem
    gidx = {_stem(n): i for i, n in enumerate(gt["names"])}
    e_i, g_i = [], []
    for j, n in enumerate(est["names"]):
        s = _stem(n)
        if s in gidx:
            e_i.append(j); g_i.append(gidx[s])
    C = est["centers"][e_i]; R = est["R_c2w"][e_i]
    Cg = gt["centers"][g_i]; Rg = gt["R_c2w"][g_i]

    res = []
    for W in all_signed_perms():
        Cw = (W @ C.T).T
        Rw = np.array([W @ r for r in R])
        r = evaluate_poses(Cw, Rw, Cg, Rg, scene_extent=extent)
        res.append((r["rot_mean_deg"], r["trans_rmse_pct"],
                    int(round(np.linalg.det(W))), W))
    res.sort(key=lambda x: x[0])

    print("best 3 WORLD transforms (lowest rotation error):")
    for rot, tpct, det, W in res[:3]:
        print(f"  rot_mean={rot:8.3f}°  trans={tpct:6.2f}%  det={det:+d}")
        print(np.array2string(W, prefix="    "))
    print(f"\nworst: rot_mean={res[-1][0]:.3f}°")


if __name__ == "__main__":
    main()