# Advanced 3D Gaussian Splatting

**COLMAP 없이 3D Gaussian Splatting 학습하기: Depth Anything 3 + gsplat**

3DGS 학습에는 보통 COLMAP SfM이 만든 카메라 파라미터(K, R|t)와 초기 sparse point가 필요합니다. 이 프로젝트는 그 전처리를 3D foundation model인 [Depth Anything 3 (DA3)](https://github.com/ByteDance-Seed/Depth-Anything-3)의 feed-forward 추론 한 번으로 대체하고, 그 출력을 COLMAP 바이너리 포맷으로 변환해 [gsplat](https://github.com/nerfstudio-project/gsplat)에 그대로 넣는 파이프라인을 실험한 기록입니다. gsplat 코드는 수정하지 않았습니다.

```
기존   images ─▶ COLMAP (feature · match · SfM) ─▶ K, R|t, sparse points ─▶ gsplat (30k step)
제안   images ─▶ DA3 (feed-forward 1회)          ─▶ K, R|t, depth → points ─▶ COLMAP 포맷(.bin) ─▶ gsplat (30k step)
```

| 장면 | 이미지 수 | COLMAP → gsplat | **DA3 → gsplat (COLMAP-free)** | 차이 |
|---|---:|---:|---:|---:|
| Truck (Tanks&Temples) | 251 | 25.22 dB | **20.28 dB** | −4.94 dB |
| Train (Tanks&Temples) | 301 | 21.94 dB | **18.30 dB** | −3.64 dB |

test PSNR, 30k step, `test_every=8` (Truck 32장, Train 38장 held-out).

---

## 데모 영상

두 모델 모두 같은 gsplat 설정(`default` strategy, 30k step)으로 학습했습니다. 왼쪽은 COLMAP 입력, 오른쪽은 DA3 출력만으로 학습한 결과입니다. 궤적은 각 모델의 카메라 좌표계에서 생성했기 때문에 두 영상의 프레임이 정확히 같은 시점은 아닙니다.

**Truck** · 왼쪽 COLMAP (PSNR 25.22), 오른쪽 DA3 COLMAP-free (PSNR 20.28)

![Truck: COLMAP vs DA3](assets/demo_truck.gif)

**Train** · 왼쪽 COLMAP (PSNR 21.94), 오른쪽 DA3 COLMAP-free (PSNR 18.30)

![Train: COLMAP vs DA3](assets/demo_train.gif)

원본 화질 영상: [Truck](assets/videos/truck_colmap_vs_da3.mp4) · [Train](assets/videos/train_colmap_vs_da3.mp4)

---

## 같은 시점에서 본 각 설정의 결과

gsplat의 test view #5를 각 설정으로 렌더링한 결과입니다(30k step). 설정 이름 아래의 `K / 포즈 / 포인트`는 각 입력을 COLMAP과 DA3 중 어디서 가져왔는지 나타냅니다.

### Truck

| 정답 (GT) | ① COLMAP 기준선 |
|:---:|:---:|
| ![GT](assets/compare/truck_v5_gt.jpg) | ![COLMAP](assets/compare/truck_single_v5.jpg) |
| test view #5 | K: COLMAP / 포즈: COLMAP / 포인트: COLMAP<br>**PSNR 25.22 · SSIM 0.879 · LPIPS 0.098** |

| ② DA3 포즈 | ③ DA3 포즈 + DA3 포인트 (conf 필터) |
|:---:|:---:|
| ![DA3 pose](assets/compare/truck_da3pose_v5.jpg) | ![DA3 pose + points filtered](assets/compare/truck_da3full_136k_v5.jpg) |
| K: COLMAP / 포즈: **DA3** / 포인트: COLMAP<br>**PSNR 21.05 · SSIM 0.688 · LPIPS 0.205** | K: COLMAP / 포즈: **DA3** / 포인트: **DA3** (confidence 필터)<br>**PSNR 17.19 · SSIM 0.592 · LPIPS 0.345** |

| ④ DA3 포즈 + DA3 포인트 (필터 없음) | ⑤ COLMAP-free (DA3 전체) |
|:---:|:---:|
| ![DA3 pose + points](assets/compare/truck_da3nofilter_136k_v5.jpg) | ![COLMAP-free](assets/compare/truck_colmapfree_136k_v5.jpg) |
| K: COLMAP / 포즈: **DA3** / 포인트: **DA3**<br>**PSNR 20.25 · SSIM 0.662 · LPIPS 0.239** | K: **DA3** / 포즈: **DA3** / 포인트: **DA3**<br>**PSNR 20.28 · SSIM 0.662 · LPIPS 0.241** |

### Train

| 정답 (GT) | COLMAP 기준선 | COLMAP-free (DA3 전체) |
|:---:|:---:|:---:|
| ![GT](assets/compare/train_v5_gt.jpg) | ![COLMAP](assets/compare/train_colmap_v5.jpg) | ![COLMAP-free](assets/compare/train_colmapfree_182k_v5.jpg) |
| test view #5 | **PSNR 21.94 · SSIM 0.811 · LPIPS 0.162** | **PSNR 18.30 · SSIM 0.584 · LPIPS 0.331** |

수치는 각 설정의 test 전체 평균이며, 이미지 한 장에 대한 값이 아닙니다.

---

## 정량 결과 (Truck ablation)

| # | 설정 | K | 포즈 | 포인트 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Gaussians | PSNR @7k |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ① | COLMAP 기준선 (`truck_single`) | COLMAP | COLMAP | COLMAP | **25.22** | **0.879** | **0.098** | 2.55M | 23.76 |
| ② | DA3 포즈 (`truck_da3pose`) | COLMAP | DA3 | COLMAP | 21.05 | 0.688 | 0.205 | 2.18M | 20.49 |
| ③ | DA3 포즈 + 포인트, conf 필터 (`truck_da3full_136k`) | COLMAP | DA3 | DA3 | 17.19 | 0.592 | 0.345 | 0.71M | 17.03 |
| ④ | DA3 포즈 + 포인트 (`truck_da3nofilter_136k`) | COLMAP | DA3 | DA3 | 20.25 | 0.662 | 0.239 | 1.99M | 19.64 |
| ⑤ | **COLMAP-free** (`truck_colmapfree_136k`) | DA3 | DA3 | DA3 | 20.28 | 0.662 | 0.241 | 1.94M | 19.71 |

- 품질 손실의 대부분은 **포즈 교체(① → ②, −4.17 dB)** 에서 생깁니다. intrinsic과 초기 포인트까지 DA3로 바꿔도(② → ⑤) 추가 손실은 0.77 dB입니다.
- ④와 ⑤가 거의 같아서, intrinsic을 DA3 예측값으로 바꿔도 손실은 거의 없었습니다.
- 초기 포인트는 confidence 필터를 쓰지 않은 쪽(④)이 필터를 쓴 쪽(③)보다 3.06 dB 높았습니다. Gaussian 수도 0.71M에서 1.99M으로 늘었습니다. 원인은 아직 분석하지 않았습니다.
- DA3 포인트 수는 비교를 위해 COLMAP sparse point 수와 맞췄습니다(Truck 136,029개, Train 182,686개).
- 2-GPU로 학습한 COLMAP 기준선(`truck`)은 PSNR 23.73으로, 단일 GPU 결과(①)를 기준으로 썼습니다.

원본 수치: [`results/gsplat/*/val_step29999.json`](results/gsplat)

---

## 실험 과정

### 1. 환경 구축과 기준선 (7/22–23)
- Docker 컨테이너 두 개(`gsplat`, `da3`)를 구성하고, Truck에서 COLMAP 입력으로 gsplat을 학습해 기준선을 잡았습니다.
- DA3 공식 예제를 돌려 출력(extrinsic, intrinsic, depth, conf, Gaussian)의 구조와 좌표계를 확인했습니다. → `scripts/01_da3_feedforward/control.py`, `diag.py`, `sweep.py`, `sanity_check*.py`

### 2. DA3 feed-forward 3DGS를 그대로 쓰기 (7/29–31)
DA3의 GS head가 바로 내놓는 Gaussian을 pose-free로 렌더링해 held-out 뷰에서 평가했습니다. 입력 뷰를 2장에서 128장까지 늘려도 PSNR이 12 dB 근처에서 멈췄고, VRAM은 뷰 수에 비례해 늘어 180장에서 OOM이 났습니다(RTX 5090 32GB). 그대로는 장면 단위 복원에 쓰기 어렵다고 보고, DA3를 **학습 기반 gsplat의 입력 생성기**로 쓰는 방향으로 바꿨습니다.

| 입력 뷰 | 2 | 8 | 24 | 48 | 128 | 180 |
|---|---:|---:|---:|---:|---:|---:|
| PSNR (dB) | 10.05 | 10.19 | 11.72 | 11.94 | 11.66 | OOM |
| peak VRAM (GB) | 7.4 | 9.3 | 10.6 | 14.1 | 25.8 | – |

<img src="assets/da3_feedforward.gif" width="480" alt="DA3 feed-forward Gaussian (24 views, pose-free)">

→ `scripts/01_da3_feedforward/view_scaling_eval.py`, `eval_posefree.py`, `posefree_24.py`, 결과 [`results/da3_feedforward/`](results/da3_feedforward)

### 3. 포즈만 DA3로 교체 (7/31)
Truck 251장 전체를 DA3에 한 번에 넣어 포즈를 예측하고, Umeyama Sim(3)로 COLMAP 좌표계에 정렬한 뒤 `images.bin`만 교체했습니다. → `predict_all_poses.py`, `make_da3_colmap.py` (설정 ②)

### 4. 초기 포인트도 DA3 depth로 교체 (8/1–8/6)
DA3 depth를 뷰별로 unprojection해 COLMAP과 같은 개수의 포인트를 만들어 `points3D.bin`을 교체했습니다. → `extract_da3_geometry.py`, `make_da3_points.py` (설정 ③), `make_da3_points_nofilter.py` (설정 ④)

### 5. 완전한 COLMAP-free 파이프라인 (8/6–8/7)
intrinsic까지 DA3 예측값을 원본 해상도로 환산해 쓰고, Sim(3) 정렬 없이 DA3 좌표계 그대로 `cameras.bin`, `images.bin`, `points3D.bin`을 직접 작성했습니다. Train 장면에서도 같은 방식으로 재현했습니다. → `make_da3_full_colmapfree.py`, `make_da3_full_colmapfree_train.py` (설정 ⑤)

### 6. 포즈 정확도와 속도 비교, Statue (8/27–28)
GT 카메라가 있는 합성 장면 Statue에서 12 · 24 · 50장으로 COLMAP(pycolmap)과 DA3 포즈를 비교했습니다. GT와는 Umeyama로 정렬했습니다.

| 방법 | 뷰 | 시간 (s) | 위치 RMSE (% of scene) | 회전 오차 평균 (°) |
|---|---:|---:|---:|---:|
| COLMAP | 12 | 2.80 | 0.08 | 0.05 |
| DA3 (ray) | 12 | 1.20 | 28.76 | 171.20 |
| COLMAP | 24 | 7.35 | 0.06 | 0.03 |
| DA3 (ray) | 24 | 1.74 | 31.27 | 170.90 |
| COLMAP | 50 | 27.52 | 0.09 | 0.04 |
| DA3 (ray) | 50 | 3.04 | 30.66 | 170.51 |

DA3는 뷰가 늘어도 수 초 안에 끝나지만, 회전 오차가 약 171°로 일정하게 나와 GT와의 좌표계 정렬(축 방향 규약)에 문제가 있는 것으로 보입니다. **아직 해결하지 못한 상태**입니다. → `scripts/03_pose_eval/`, 결과 [`results/statue_pose/`](results/statue_pose)

---

## 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA RTX 5090 32GB × 2 (학습은 단일 GPU) |
| Docker 이미지 | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` (두 컨테이너 공통, `--gpus all --shm-size 16g`) |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) @ `bf302ef5`, 수정 없음 |
| Depth Anything 3 | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) @ `3fe327a` + [`patches/depth-anything-3.patch`](patches/depth-anything-3.patch) |
| DA3 모델 | `DA3NESTED-GIANT-LARGE-1.1`, `process_res=480`, `upper_bound_resize` |
| 데이터 | Tanks&Temples Truck / Train (3DGS 배포본), Statue (합성, GT 카메라 포함) |

DA3 패치는 CLI의 `reference_view_strategy` 인자 이름 불일치와 최신 Matplotlib에서 제거된 `cm.get_cmap` 호출을 고친 두 줄짜리 수정입니다.

컨테이너 구성은 다음과 같습니다. 두 컨테이너의 마운트가 분리되어 있어서, DA3에서 만든 데이터 폴더는 `gsplat_workspace/data/`로 복사해 학습했습니다.

| 컨테이너 | 호스트 경로 → `/workspace` | 용도 |
|---|---|---|
| `da3` | `~/da3_workspace` | DA3 추론, COLMAP 포맷 변환 (`scripts/01`, `02`, `03/run_da3_pose.py`) |
| `gsplat` | `~/gsplat_workspace` | COLMAP, gsplat 학습 (`scripts/03/run_colmap.py`) |
| 호스트 | – | 포즈 비교 (`compare_poses.py`, `pose_eval.py`) |

## 재현 방법 (Truck, COLMAP-free)

스크립트 안의 경로는 컨테이너 기준(`/workspace/...`)으로 작성되어 있습니다.

```bash
# [da3 컨테이너] 1) 251장 전체를 DA3로 추론해 depth / conf / K / R|t 저장 → /workspace/results/da3_geometry.npz
python scripts/02_da3_to_colmap/extract_da3_geometry.py

# [da3 컨테이너] 2) DA3 출력만으로 COLMAP 바이너리 작성 (인자: 초기 포인트 수)
python scripts/02_da3_to_colmap/make_da3_full_colmapfree.py 136029
#   → /workspace/data/truck_colmapfree_136k/sparse/0/{cameras,images,points3D}.bin

# [호스트] 3) gsplat 쪽으로 데이터 복사
cp -r ~/da3_workspace/data/truck_colmapfree_136k ~/gsplat_workspace/data/

# [gsplat 컨테이너] 4) 학습 (gsplat/examples)
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --data_dir /workspace/data/truck_colmapfree_136k \
  --data_factor 1 --test_every 8 \
  --result_dir ./results/truck_colmapfree_136k \
  --disable_viewer --save_ply
```

`extract_da3_geometry.py`는 이미지 목록을 읽을 때와 이전 단계 비교용으로 원본 데이터셋의 COLMAP 폴더를 읽습니다. 하지만 `make_da3_full_colmapfree.py`가 쓰는 카메라·포인트에는 COLMAP 값이 들어가지 않습니다.

## 레포 구조

```
assets/                 README용 GIF, 비교 이미지(test view #5), mp4
patches/                Depth-Anything-3 로컬 수정 사항
results/
  gsplat/<run>/         각 학습의 cfg.yml, val_step6999.json, val_step29999.json
  da3_feedforward/      뷰 수에 따른 feed-forward GS 품질 · VRAM
  statue_pose/          COLMAP vs DA3 포즈 비교, 단계별 시간
scripts/
  01_da3_feedforward/   DA3 출력 확인, feed-forward GS 평가 (실험 과정 1–2)
  02_da3_to_colmap/     DA3 → COLMAP 포맷 변환 파이프라인 (실험 과정 3–5)
  03_pose_eval/         Statue 포즈 정확도 · 시간 비교 (실험 과정 6)
```

체크포인트, `.ply`, 데이터셋은 용량 때문에 포함하지 않았습니다.

## 다음 단계

- gsplat의 `pose_opt`를 켜서 DA3 포즈 오차를 학습 중에 보정하기
- Statue 포즈 비교의 좌표계 정렬 문제 해결
- Truck 251장 기준으로 COLMAP 전처리와 DA3 추론의 end-to-end 시간 비교
