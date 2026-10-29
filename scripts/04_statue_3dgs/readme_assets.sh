set -e
F=/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
R=/workspace/statue3dgs/results; O=/workspace/statue3dgs/readme; mkdir -p $O
python - <<'PY'
from PIL import Image
R='/workspace/statue3dgs/results'; O='/workspace/statue3dgs/readme/'
for v in ('0026','0051'):
    Image.open(f'/workspace/data/Statue/fs/{v}.jpg').convert('RGB').resize((640,360),Image.LANCZOS).save(f'{O}statue_{v}_gt.jpg',quality=85)
    for N in (12,24,50):
        for m in ('colmap','da3_decoder','da3_ray'):
            Image.open(f'{R}/{m}_{N}/compare/{v}_tto.jpg').resize((640,360),Image.LANCZOS).save(f'{O}statue_{v}_{m}_{N}.jpg',quality=85)
PY
for N in 24 50; do
  $F -y -loglevel error -i $R/colmap_$N/videos/traj_29999.mp4 -i $R/da3_decoder_$N/videos/traj_29999.mp4 -i $R/da3_ray_$N/videos/traj_29999.mp4 \
    -filter_complex "[0:v]crop=iw/2:ih:0:0,scale=426:-2[a];[1:v]crop=iw/2:ih:0:0,scale=426:-2[b];[2:v]crop=iw/2:ih:0:0,scale=426:-2[c];[a][b][c]hstack=3" \
    -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p -an -movflags +faststart $O/statue_${N}_3way.mp4
  $F -y -loglevel error -i $O/statue_${N}_3way.mp4 -vf "fps=10,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" $O/statue_${N}_3way.gif
done
ls -la $O | awk '{print $5, $9}'
