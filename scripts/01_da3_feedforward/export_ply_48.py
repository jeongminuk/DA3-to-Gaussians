import os, numpy as np
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler

COLMAP_DIR = "/workspace/data/tandt/truck"
MODEL_DIR  = "/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
N_CONTEXT  = 96 # 입력 뷰 수 조절
PROCESS_RES = 480

# COLMAP 로드 + 파일명 정렬 (평가 때와 동일)
image_files, extr, intr = ColmapHandler.process(COLMAP_DIR, "0")
order = np.argsort([os.path.basename(f) for f in image_files])
image_files = [image_files[i] for i in order]
extr, intr = np.asarray(extr)[order], np.asarray(intr)[order]
n = len(image_files)

# test(8번째마다) 제외한 pool에서 균등 48장 (평가 스크립트와 동일 규칙)
test_idx = np.arange(0, n, 8)
train_pool = np.setdiff1d(np.arange(n), test_idx)
ctx_idx = train_pool[np.linspace(0, len(train_pool)-1, N_CONTEXT).round().astype(int)]
print(f"context {N_CONTEXT}장: {ctx_idx.tolist()}")

model = DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred = model.inference(
    image=[image_files[i] for i in ctx_idx],
    extrinsics=extr[ctx_idx], intrinsics=intr[ctx_idx],
    infer_gs=True,
    align_to_input_ext_scale=False,
    process_res=PROCESS_RES, process_res_method="upper_bound_resize",
    export_dir="/workspace/results/truck_ply_48",
    export_format="gs_ply",
    export_kwargs={"gs_ply": {"gs_views_interval": 1}},
)
print("done")
for root,_,fs in os.walk("/workspace/results/truck_ply_48"):
    for f in fs:
        p = os.path.join(root,f)
        print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")