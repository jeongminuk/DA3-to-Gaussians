"""
run_da3_pose.py   (run inside the da3 container)
Runs DA3 inference on a folder of images and saves the predicted camera
extrinsics for pose evaluation. DA3's pose is an OPTIONAL output
(output.extrinsics); we pull it straight from the Prediction object.

IMPORTANT (see da3.py):
  - camera_decoder path (use_ray_pose=False, default):
        output.extrinsics = affine_inverse(c2w)  => stored as W2C
  - ray path (use_ray_pose=True):
        output.extrinsics is already C2W
  We just save the raw tensor + the source tag; pose_eval.load_da3_poses
  converts to C2W using the matching `source`.

Example:
  python run_da3_pose.py \
      --image_dir /workspace/data/Statue/fs/train_12 \
      --out_dir   /workspace/exp/Statue/fs/da3_12 \
      --model_dir /workspace/models/DA3NESTED-GIANT-LARGE-1.1
"""
import argparse, glob, json, os, time
import numpy as np


IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_dir",
                    default="/workspace/models/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_ray_pose", action="store_true",
                    help="use ray-based pose instead of camera decoder")
    ap.add_argument("--ref_view_strategy", default="saddle_balanced")
    ap.add_argument("--process_res", type=int, default=504)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs = sorted(f for f in glob.glob(os.path.join(args.image_dir, "*"))
                  if f.lower().endswith(IMG_EXT))
    if not imgs:
        raise RuntimeError(f"no images in {args.image_dir}")

    # Import here so --help works without the heavy deps loaded.
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(args.model_dir).to(args.device)

    t0 = time.perf_counter()
    pred = model.inference(
        image=imgs,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        process_res=args.process_res,
        export_dir=None,             # we only want the poses, no GLB
    )
    dt = time.perf_counter() - t0

    if getattr(pred, "extrinsics", None) is None:
        raise RuntimeError("prediction.extrinsics is None (no camera output).")

    ext = pred.extrinsics
    ext = ext.detach().cpu().numpy() if hasattr(ext, "detach") else np.asarray(ext)
    ext = np.squeeze(ext)            # (1,N,4,4) -> (N,4,4) if batched
    if ext.ndim != 3:
        raise RuntimeError(f"unexpected extrinsics shape {ext.shape}")

    source = "ray" if args.use_ray_pose else "camera_decoder"
    np.save(os.path.join(args.out_dir, "pred_extrinsics.npy"), ext)
    with open(os.path.join(args.out_dir, "names.txt"), "w") as f:
        f.write("\n".join(os.path.basename(p) for p in imgs))

    timing = {"n_views": len(imgs), "seconds": round(dt, 3),
              "source": source, "ref_view_strategy": args.ref_view_strategy,
              "shape": list(ext.shape), "model_dir": args.model_dir}
    with open(os.path.join(args.out_dir, "da3_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()