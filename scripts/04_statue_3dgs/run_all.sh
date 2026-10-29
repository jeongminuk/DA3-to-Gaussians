# usage: bash run_all.sh <gpu> <run> [<run> ...]   e.g. colmap_12
cd /workspace/gsplat/examples
GPU=$1; shift
for R in "$@"; do
  [ -f /workspace/statue3dgs/results/$R/stats/val_step29999.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU python /workspace/statue3dgs/train_split.py default \
    --data_dir /workspace/statue3dgs/data/$R --data_factor 1 \
    --result_dir /workspace/statue3dgs/results/$R --disable_viewer \
    > /workspace/statue3dgs/logs/$R.log 2>&1
  echo "$R done $(date +%T)"
done
