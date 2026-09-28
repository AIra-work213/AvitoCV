#!/usr/bin/env python3
"""Mine continuous long text lines from neighboring Yandex OCR polygons."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from prepare_yandex_ocr_realistic_validation import curve_upright, guarded_degrade, image_stats, make_contact_sheet


def boxes(path: Path):
    out=[]
    for line_no,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        p=line.split(',',8)
        if len(p)!=9 or not p[8].strip() or p[8].strip()=='###': continue
        q=np.array([float(x) for x in p[:8]],np.float32).reshape(4,2)
        w=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2
        h=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
        if w<h or h<3: continue
        direction=(q[1]-q[0])+(q[2]-q[3]); direction/=max(np.linalg.norm(direction),1e-6)
        if direction[0]<0: direction=-direction
        out.append(dict(points=q,width=w,height=h,direction=direction,center=q.mean(0),text=p[8].strip(),line=line_no))
    return out


def components(items):
    n=len(items); parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for i in range(n):
        for j in range(i+1,n):
            a,b=items[i],items[j]
            angle=math.acos(float(np.clip(a['direction']@b['direction'],-1,1)))
            if angle>math.radians(12) or max(a['height'],b['height'])/min(a['height'],b['height'])>1.8: continue
            u=a['direction']+b['direction']; u/=np.linalg.norm(u); v=np.array([-u[1],u[0]])
            delta=b['center']-a['center']; gap=abs(delta@u)-(a['width']+b['width'])/2
            if -.35*max(a['width'],b['width'])<=gap<=4*max(a['height'],b['height']) and abs(delta@v)<=.65*max(a['height'],b['height']): union(i,j)
    result={}
    for i in range(n): result.setdefault(find(i),[]).append(i)
    return [x for x in result.values() if len(x)>=2]


def rectify_line(image, group):
    u=np.mean([x['direction'] for x in group],axis=0); u/=np.linalg.norm(u)
    v=np.array([-u[1],u[0]],np.float32)
    down=np.mean([((x['points'][2]+x['points'][3])-(x['points'][0]+x['points'][1]))/2 for x in group],axis=0)
    if down@v<0: v=-v
    pts=np.concatenate([x['points'] for x in group]); pu=pts@u; pv=pts@v
    lo_u,hi_u=float(pu.min()),float(pu.max()); lo_v,hi_v=float(pv.min()),float(pv.max())
    width=max(2,round(hi_u-lo_u)); height=max(2,round(hi_v-lo_v))
    # Reject components that accidentally chained neighboring rows.
    med_h=float(np.median([x['height'] for x in group]))
    if height>2.1*med_h or width/height<6: return None
    src=np.array([u*lo_u+v*lo_v,u*hi_u+v*lo_v,u*hi_u+v*hi_v,u*lo_u+v*hi_v],np.float32)
    dst=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(src,dst),(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,default=Path('data/train/rus_ocr_in_the_wild_dataset')); p.add_argument('--test-images',type=Path,default=Path('data/test/test/images')); p.add_argument('--output',type=Path,default=Path('data/train/yandex_ocr_real_long_lines_v4')); p.add_argument('--samples',type=int,default=553); p.add_argument('--seed',type=int,default=20260926); a=p.parse_args()
    candidates=[]
    for ann in sorted(a.source.glob('gt_img_*.txt')):
        image_path=a.source/(ann.name[3:-4]+'.png'); image=cv2.imread(str(image_path))
        if image is None: continue
        items=boxes(ann)
        for ids in components(items):
            group=[items[i] for i in ids]; crop=rectify_line(image,group)
            if crop is None or crop.shape[0]<12 or cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).std()<12.8: continue
            candidates.append(dict(crop=crop,source_image=image_path.name,source_lines=';'.join(str(x['line']) for x in group),text=' '.join(x['text'] for x in sorted(group,key=lambda x:float(x['center']@group[0]['direction']))),ratio=crop.shape[1]/crop.shape[0],components=len(group)))
    # Select natural lines whose empirical ratio quantiles best match test ratio>=6.
    test=[]
    for path in a.test_images.glob('*'):
        im=cv2.imread(str(path))
        if im is not None and im.shape[1]/im.shape[0]>=6:test.append(im.shape[1]/im.shape[0])
    desired=np.quantile(np.sort(test),(np.arange(min(a.samples,len(candidates)))+.5)/min(a.samples,len(candidates)))
    available=set(range(len(candidates))); selected=[]
    for ratio in desired:
        idx=min(available,key=lambda i:abs(math.log(candidates[i]['ratio']/ratio))); available.remove(idx); selected.append(candidates[idx])
    random.Random(a.seed).shuffle(selected)
    out=a.output/'images'; out.mkdir(parents=True,exist_ok=True); rows=[]
    rotated=set(random.Random(a.seed+1).sample(range(len(selected)),len(selected)//2))
    for i,c in enumerate(selected):
        rng=np.random.default_rng(a.seed+i); image=c.pop('crop'); original=image.copy(); degradation='clean'; corr=1.0; x=rng.random()
        if x<.10: degradation='hard_readable'; image,corr=guarded_degrade(image,rng,degradation)
        elif x<.35: degradation='moderate'; image,corr=guarded_degrade(image,rng,degradation)
        # Never retain the rare failed degradation that makes the text disappear.
        if corr < .65 or cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).std() < 12.8:
            image=original; degradation='clean_fallback'; corr=1.0
        geometry='straight'
        if rng.random()<.12 and image.shape[1]/image.shape[0]>=8: geometry='curved'; image=curve_upright(image,rng)
        target=int(i in rotated)
        if target:image=cv2.rotate(image,cv2.ROTATE_180)
        image_id=f'yandex_real_line_{i:05d}'; path=out/(image_id+'.png'); cv2.imwrite(str(path),image); s=image_stats(image)
        rows.append(dict(image_id=image_id,image_path=str(path),target=target,**c,degradation=degradation,geometry=geometry,structural_correlation=f'{corr:.6f}',output_width=s['width'],output_height=s['height'],output_ratio=f"{s['ratio']:.6f}",output_std=f"{s['std']:.6f}",output_laplacian_variance=f"{s['laplacian_variance']:.6f}"))
    df=pd.DataFrame(rows); df.to_csv(a.output/'labels.csv',index=False); make_contact_sheet(rows,a.output/'contact_sheet.jpg'); r=df.output_ratio.astype(float).to_numpy()
    summary={'candidates':len(candidates),'samples':len(df),'construction':'continuous perspective crop from neighboring polygons in one source photo; no stitching','ratio_counts':{f'ge_{x}':int((r>=x).sum()) for x in [6,7,8,9,10,12,15,20]},'ratio_quantiles':dict(zip(['min','p10','p25','p50','p75','p90','p95','p99','max'],map(float,np.quantile(r,[0,.1,.25,.5,.75,.9,.95,.99,1])))),'degradation':df.degradation.value_counts().to_dict(),'geometry':df.geometry.value_counts().to_dict()}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
