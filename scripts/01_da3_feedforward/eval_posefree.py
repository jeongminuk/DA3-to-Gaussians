import os, numpy as np, torch, imageio
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import ColmapHandler
from depth_anything_3.model.utils.gs_renderer import render_3dgs
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

COLMAP_DIR="/workspace/data/tandt/truck"; MODEL_DIR="/workspace/models/DA3NESTED-GIANT-LARGE-1.1"
OUT="/workspace/results/eval_posefree"; N_CTX=24; TEST_EVERY=8; PRES=480
os.makedirs(OUT, exist_ok=True)

image_files, extr, intr = ColmapHandler.process(COLMAP_DIR, "0")
o = np.argsort([os.path.basename(f) for f in image_files])
image_files=[image_files[i] for i in o]; extr=np.asarray(extr)[o]; intr=np.asarray(intr)[o]
n=len(image_files)
test_idx=np.arange(0,n,TEST_EVERY); pool=np.setdiff1d(np.arange(n),test_idx)
ctx_idx=pool[np.linspace(0,len(pool)-1,N_CTX).round().astype(int)]
print(f"total {n} | ctx {N_CTX} | test {len(test_idx)}")

model=DepthAnything3.from_pretrained(MODEL_DIR).to("cuda")
pred=model.inference(                       # 포즈 안 넘김 = pose-free
    image=[image_files[i] for i in ctx_idx],
    infer_gs=True, process_res=PRES, process_res_method="upper_bound_resize")
g=pred.gaussians; H,W=pred.processed_images.shape[1:3]
print(f"{H}x{W} | GS {g.means.shape[1]:,}")

def to4(e):
    e=np.asarray(e)
    return e if e.shape[-2]==4 else np.concatenate(
        [e, np.tile(np.array([[[0,0,0,1.]]]),(e.shape[0],1,1))],axis=1)
def cen(E):
    R,t=E[:,:3,:3],E[:,:3,3]; return np.einsum("nij,nj->ni",R.transpose(0,2,1),-t)
def umeyama(s,d):
    ms,md=s.mean(0),d.mean(0); S,D=s-ms,d-md
    C=D.T@S/len(s); U,sig,Vt=np.linalg.svd(C); F=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: F[2,2]=-1
    R=U@F@Vt; sc=(sig*np.diag(F)).sum()/(S**2).sum()*len(s)
    return sc,R,md-sc*R@ms

da3_c=to4(pred.extrinsics); col_c=to4(extr[ctx_idx])
sc,Ra,ta=umeyama(cen(col_c),cen(da3_c))
res=np.linalg.norm((sc*(Ra@cen(col_c).T).T+ta)-cen(da3_c),axis=1)
scene=np.linalg.norm(cen(da3_c).max(0)-cen(da3_c).min(0))
print(f"\n=== 포즈 정확도 (DA3 예측 vs COLMAP) ===")
print(f"Sim3 scale {sc:.5f}")
print(f"위치 오차: mean {res.mean():.4f}, max {res.max():.4f} (씬 크기 {scene:.3f} 대비 {100*res.mean()/scene:.2f}%)")
Rrel=np.einsum("nij,njk->nik", da3_c[:,:3,:3], (col_c[:,:3,:3]@Ra.T).transpose(0,2,1))
ang=np.degrees(np.arccos(np.clip((np.trace(Rrel,axis1=1,axis2=2)-1)/2,-1,1)))
print(f"회전 오차: mean {ang.mean():.3f}°, max {ang.max():.3f}°")

def c2d(E):
    E=to4(E); Rd=E[:,:3,:3]@Ra.T
    td=sc*E[:,:3,3]-np.einsum("nij,j->ni",Rd,ta)
    out=np.tile(np.eye(4),(len(E),1,1)); out[:,:3,:3],out[:,:3,3]=Rd,td; return out

psnr_fn=PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_fn=StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_fn=LearnedPerceptualImagePatchSimilarity(net_type="alex",normalize=True).cuda()

def ev(idxs,tag):
    te=torch.from_numpy(c2d(extr[idxs])).float().cuda()
    K=torch.from_numpy(np.asarray(pred.intrinsics)[0]).float().cuda().unsqueeze(0).repeat(len(idxs),1,1)
    Kn=K.clone(); Kn[:,0,:]/=W; Kn[:,1,:]/=H
    gt=torch.from_numpy(np.stack([np.asarray(Image.open(image_files[i]).convert("RGB")
        .resize((W,H),Image.LANCZOS)) for i in idxs])).float().cuda()/255.
    outs=[]
    with torch.no_grad():
        for s in range(0,len(idxs),4):
            c,_=render_3dgs(extrinsics=te[s:s+4],intrinsics=Kn[s:s+4],
                            image_shape=(H,W),gaussian=g,num_view=len(te[s:s+4])); outs.append(c)
    rd=torch.cat(outs).permute(0,2,3,1).clamp(0,1)
    P,S,L=[],[],[]
    for i in range(len(idxs)):
        a=rd[i:i+1].permute(0,3,1,2); b=gt[i:i+1].permute(0,3,1,2)
        P.append(psnr_fn(a,b).item()); S.append(ssim_fn(a,b).item()); L.append(lpips_fn(a,b).item())
    for i in range(min(3,len(idxs))):
        imageio.imwrite(f"{OUT}/{tag}_render_{i}.png",(rd[i].cpu().numpy()*255).astype(np.uint8))
        imageio.imwrite(f"{OUT}/{tag}_gt_{i}.png",(gt[i].cpu().numpy()*255).astype(np.uint8))
    print(f"{tag:8s} PSNR {np.mean(P):7.3f}  SSIM {np.mean(S):.4f}  LPIPS {np.mean(L):.4f}")
    return np.mean(P),np.mean(S),np.mean(L)

print(f"\n=== NVS 품질 (pose-free) ===")
ev(ctx_idx,"context"); ev(test_idx,"test")
print(f"\nGS {g.means.shape[1]:,} | 저장 {OUT}/")