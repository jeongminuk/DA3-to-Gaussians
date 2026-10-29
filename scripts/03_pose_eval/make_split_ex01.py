"""
make_split.py
Deterministic train/test splits for IVL-SYNTHSFM-v2 scenes.
Uniformly subsamples N train views from the 100; the rest are test.
Same split is reused for pose eval AND novel-view rendering (PSNR).

Usage:
  python make_split.py --csv /workspace/data/Statue/fs/cameras.csv \
                       --out /workspace/data/Statue/fs/splits.json \
                       --sizes 12 24 50
Optional: --images-dir DIR --ext .png  -> also builds symlink folders
          train_<N>/ and test_<N>/ next to the csv for COLMAP input.
"""
import argparse, csv, json, os
import numpy as np


def read_names(cameras_csv):
    with open(cameras_csv, newline="") as f:
        return [r["image_number"] for r in csv.DictReader(f)]


def uniform_train_idx(n_total, n_train):
    """Evenly spaced indices across the ordered trajectory."""
    idx = np.linspace(0, n_total - 1, n_train).round().astype(int)
    return sorted(set(idx.tolist()))


def build_splits(names, sizes):
    n = len(names)
    splits = {}
    for k in sizes:
        tr = uniform_train_idx(n, k)
        te = [i for i in range(n) if i not in set(tr)]
        splits[str(k)] = {
            "train": [names[i] for i in tr],
            "test":  [names[i] for i in te],
            "train_idx": tr, "test_idx": te,
        }
    return splits


def materialize(splits, images_dir, ext, base_dir):
    """Create train_<N>/ test_<N>/ folders of symlinks for COLMAP."""
    for k, sp in splits.items():
        for part in ("train", "test"):
            d = os.path.join(base_dir, f"{part}_{k}")
            os.makedirs(d, exist_ok=True)
            for name in sp[part]:
                src = os.path.join(images_dir, f"{name}{ext}")
                dst = os.path.join(d, f"{name}{ext}")
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 24, 50])
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    names = read_names(args.csv)
    splits = build_splits(names, args.sizes)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"wrote {args.out}  ({len(names)} views)")
    for k, sp in splits.items():
        print(f"  N={k:>2}: train={len(sp['train'])} test={len(sp['test'])}"
              f"  train ids e.g. {sp['train'][:4]}...")

    if args.images_dir:
        base = os.path.dirname(os.path.abspath(args.out))
        materialize(splits, args.images_dir, args.ext, base)
        print(f"  symlink folders train_<N>/ test_<N>/ created under {base}")