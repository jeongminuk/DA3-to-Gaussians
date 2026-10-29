import os, numpy as np
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"

image_files, extrinsics, intrinsics = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extrinsics = np.asarray(extrinsics)[order]
intrinsics = np.asarray(intrinsics)[order]

idx = np.linspace(0, len(image_files) - 1, 12).round().astype(int)

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=[image_files[i] for i in idx],
    extrinsics=extrinsics[idx],
    intrinsics=intrinsics[idx],
    infer_gs=True,
    export_dir="/workspace/results/official_render",
    export_format="gs_video-gs_ply",
    export_kwargs={
        "gs_video": {
            "trj_mode": "original",   # 입력 뷰 그대로 렌더링
            "chunk_size": 4,
            "vis_depth": None,        # RGB만
            "enable_tqdm": True,
        }
    },
)
print("done")
for root, _, files in os.walk("/workspace/results/official_render"):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")