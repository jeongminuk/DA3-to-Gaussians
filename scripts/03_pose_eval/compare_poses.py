"""
compare_poses.py   (run on the HOST -- it can read both workspaces)
Builds the Experiment #1 table: COLMAP vs DA3 pose accuracy & runtime,
aligned to GT with Umeyama (same alignment DA3's own bench uses).

COLMAP and DA3 run in separate containers with separate mounts, so their
outputs live under different host dirs. Point one flag at each:
  colmap -> gsplat_workspace/exp/Statue/fs/colmap_<N>/sparse/0/ + timing.json
  da3    -> da3_workspace/exp/Statue/fs/da3_<N>/pred_extrinsics.npy + names.txt + da3_timing.json

Example (on host):
  python compare_poses.py \
      --cameras_csv /home/minuk0221/gsplat_workspace/data/Statue/fs/cameras.csv \
      --colmap_exp  /home/minuk0221/gsplat_workspace/exp/Statue/fs \
      --da3_exp     /home/minuk0221/da3_workspace/exp/Statue/fs \
      --sizes 12
"""
import argparse, json, os
import numpy as np
from pose_eval import (load_gt_poses, load_colmap_recon, load_da3_poses,
                       match_and_evaluate, GL2CV)


def _read_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras_csv", required=True)
    ap.add_argument("--colmap_exp", required=True,
                    help="dir holding colmap_<N>/ (e.g. gsplat_workspace/exp/Statue/fs)")
    ap.add_argument("--da3_exp", required=True,
                    help="dir holding da3_<N>/ (e.g. da3_workspace/exp/Statue/fs)")
    ap.add_argument("--out_dir", default=None,
                    help="where to write comparison.json (default: da3_exp)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 24, 50])
    ap.add_argument("--da3_source", default="camera_decoder",
                    choices=["camera_decoder", "ray"])
    ap.add_argument("--da3_suffix", default="",
                    help="suffix on da3 folders, e.g. '_ray' -> da3_<N>_ray")
    args = ap.parse_args()

    gt = load_gt_poses(args.cameras_csv)
    extent = np.linalg.norm(gt["centers"] - gt["centers"].mean(0), axis=1).max()

    rows = []
    for N in args.sizes:
        # ---- COLMAP ----
        cdir = os.path.join(args.colmap_exp, f"colmap_{N}", "sparse", "0")
        if os.path.isdir(cdir):
            est = load_colmap_recon(cdir)
            r = match_and_evaluate(est, gt, scene_extent=extent)
            t = _read_json(os.path.join(args.colmap_exp, f"colmap_{N}", "timing.json"))
            rows.append(("COLMAP", N, r, t.get("total"),
                         f"{r['n_matched']}/{N}", t.get("device", "")))

        # ---- DA3 ----
        ddir = os.path.join(args.da3_exp, f"da3_{N}{args.da3_suffix}")
        npy = os.path.join(ddir, "pred_extrinsics.npy")
        if os.path.exists(npy):
            # NOTE: the ray path's poses are NOT fixable by any axis flip
            # (verified by find_world.py: best residual ~95 deg, det=-1).
            # So we report raw ray poses as-is -- no flip -- which is the
            # honest measure of the ray path's pose quality on this data.
            est = load_da3_poses(npy, os.path.join(ddir, "names.txt"),
                                 source=args.da3_source)
            r = match_and_evaluate(est, gt, scene_extent=extent)
            t = _read_json(os.path.join(ddir, "da3_timing.json"))
            rows.append((f"DA3{args.da3_suffix}", N, r, t.get("seconds"),
                         f"{r['n_matched']}/{N}", t.get("source", "")))

    # ---- print table ----
    hdr = ("method", "N", "matched", "trans_rmse", "trans_%", "rot_mean",
           "rot_med", "time_s", "note")
    print(f"{hdr[0]:<7}{hdr[1]:>4}  {hdr[2]:>8}  {hdr[3]:>10}  {hdr[4]:>8}  "
          f"{hdr[5]:>8}  {hdr[6]:>8}  {hdr[7]:>7}  {hdr[8]}")
    print("-" * 82)
    for method, N, r, secs, matched, note in rows:
        tpct = r.get("trans_rmse_pct", float("nan"))
        secs = f"{secs:.2f}" if isinstance(secs, (int, float)) else "?"
        print(f"{method:<7}{N:>4}  {matched:>8}  {r['trans_rmse']:>10.4f}  "
              f"{tpct:>7.2f}%  {r['rot_mean_deg']:>7.3f}°  "
              f"{r['rot_median_deg']:>7.3f}°  {secs:>7}  {note}")

    # ---- save machine-readable ----
    out = []
    for method, N, r, secs, matched, note in rows:
        out.append({"method": method, "N": N, "matched": matched,
                    "trans_rmse": r["trans_rmse"],
                    "trans_rmse_pct": r.get("trans_rmse_pct"),
                    "rot_mean_deg": r["rot_mean_deg"],
                    "rot_median_deg": r["rot_median_deg"],
                    "time_s": secs, "note": note})
    out_dir = args.out_dir or args.da3_exp
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(out_dir, 'comparison.json')}")


if __name__ == "__main__":
    main()