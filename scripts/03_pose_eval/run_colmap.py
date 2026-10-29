"""
run_colmap.py   (gsplat container; uses pycolmap -- no GUI, no Qt display needed)
Sparse SfM on a train image folder, timing each stage.

Example:
  python run_colmap.py \
      --image_dir /workspace/data/Statue/fs/train_12 \
      --work_dir  /workspace/exp/Statue/fs/colmap_12

Output layout (read by gsplat's Parser / pose_eval.load_colmap_recon):
  work_dir/sparse/0/{cameras,images,points3D}.bin
  work_dir/timing.json
"""
import argparse, json, os, shutil, time
import pycolmap


def timed(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--camera_model", default="PINHOLE")   # synthetic -> no distortion
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    db = os.path.join(args.work_dir, "database.db")
    if os.path.exists(db):
        os.remove(db)                       # avoid "already exists" on re-run
    sparse = os.path.join(args.work_dir, "sparse")
    os.makedirs(sparse, exist_ok=True)

    dev = getattr(pycolmap.Device, args.device)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = args.camera_model

    timing, used_device = {}, args.device

    # 1) feature extraction (single shared camera: one intrinsic for all views)
    def _extract(device):
        pycolmap.extract_features(
            database_path=db, image_path=args.image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader, device=device)
    try:
        timing["feature"] = timed(lambda: _extract(dev))
    except Exception as e:                   # headless GPU/GL fallback -> CPU
        print(f"[extract] {args.device} failed ({e}); retrying on CPU")
        if os.path.exists(db): os.remove(db)
        used_device = "cpu"
        timing["feature"] = timed(lambda: _extract(pycolmap.Device.cpu))
        dev = pycolmap.Device.cpu

    # 2) exhaustive matching (fine for sparse view counts)
    timing["match"] = timed(lambda: pycolmap.match_exhaustive(
        database_path=db, device=dev))

    # 3) incremental mapping (SfM) -> dict{recon_id: Reconstruction}
    maps = {}
    timing["mapper"] = timed(lambda: maps.update(
        pycolmap.incremental_mapping(
            database_path=db, image_path=args.image_dir, output_path=sparse)))
    if not maps:
        raise RuntimeError("COLMAP registered 0 models (too few matches?).")

    # keep the largest model as sparse/0
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    out0 = os.path.join(sparse, "0")
    if os.path.exists(out0): shutil.rmtree(out0)
    os.makedirs(out0)
    best.write(out0)

    n_input = len([f for f in os.listdir(args.image_dir)
                   if not f.startswith(".")])
    timing["total"] = round(sum(v for k, v in timing.items()
                                if isinstance(v, float)), 3)
    timing["device"] = used_device
    timing["n_input"] = n_input
    timing["n_registered"] = best.num_reg_images()
    timing["colmap_dir"] = out0
    for k in ("feature", "match", "mapper"):
        timing[k] = round(timing[k], 3)

    with open(os.path.join(args.work_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))
    if timing["n_registered"] < n_input:
        print(f"WARNING: only {timing['n_registered']}/{n_input} images "
              f"registered (sparse-view COLMAP can drop views).")


if __name__ == "__main__":
    main()