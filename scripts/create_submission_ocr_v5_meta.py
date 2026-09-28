#!/usr/bin/env python3
"""Create the OCR-v5 meta-model submission selected on grouped Yandex OOF."""
from __future__ import annotations
import argparse, json, math, random, re
from collections import Counter
from pathlib import Path
import cv2, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=20260927; C_VALUE=0.3; TEMPERATURE=0.42
FEATURE_COLUMNS=[*range(12),18,20,21]
ALPHABET=" абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz0123456789-"

def normalize(text):
 text=str(text).lower().replace('ё','е');text=''.join(c if c in ALPHABET else ' ' for c in text)
 return re.sub(r' +',' ',text).strip()

class CharLM:
 def __init__(self,texts,alpha=.2):
  self.alpha=alpha;self.context=Counter();self.ngram=Counter();self.vocabulary=len(ALPHABET)+2
  for text in texts:
   value='^^'+normalize(text)+'$$'
   for i in range(2,len(value)):self.context[value[i-2:i]]+=1;self.ngram[value[i-2:i+1]]+=1
 def score(self,text):
  value='^^'+normalize(text)+'$$'
  if len(value)<=4:return -12.
  return float(np.mean([math.log((self.ngram[value[i-2:i+1]]+self.alpha)/(self.context[value[i-2:i]]+self.alpha*self.vocabulary)) for i in range(2,len(value))]))

def corpus():
 texts=[]
 for path in Path('data/train/rus_ocr_in_the_wild_dataset').glob('gt_img_*.txt'):
  for line in path.read_text(encoding='utf-8-sig').splitlines():
   parts=line.split(',',8)
   if len(parts)==9 and parts[8].strip()!='###':texts.append(parts[8].strip())
 return texts

def apply_temp(p,t):
 p=np.clip(p,1e-9,1-1e-9);return 1/(1+np.exp(-np.log(p/(1-p))/t))

def attach(frame,path,prefix):
 cols=['image_id','rec_score_direct','rec_score_rotated','rec_score_difference','rec_text_direct','rec_text_rotated']
 extra=pd.read_csv(path)[cols].rename(columns={c:f'{prefix}_{c}' for c in cols if c!='image_id'})
 return frame.merge(extra,on='image_id')

def add_ocr(frame,cyr,eslav,eng):return attach(attach(attach(frame,cyr,'cyr'),eslav,'eslav'),eng,'eng')

def one_model(row,prefix,lm):
 direct=normalize(getattr(row,f'{prefix}_rec_text_direct'));rotated=normalize(getattr(row,f'{prefix}_rec_text_rotated'))
 cd=sum('а'<=c<='я' for c in direct)/max(1,len(direct));cr=sum('а'<=c<='я' for c in rotated)/max(1,len(rotated))
 return [getattr(row,f'{prefix}_rec_score_difference'),getattr(row,f'{prefix}_rec_score_direct'),getattr(row,f'{prefix}_rec_score_rotated'),np.tanh((len(rotated)-len(direct))/5),lm.score(rotated)-lm.score(direct),cr-cd]

def features(frame,lm):
 out=[]
 for row in frame.itertuples():
  c=one_model(row,'cyr',lm);e=one_model(row,'eslav',lm);g=one_model(row,'eng',lm);p=np.clip(row.p_current,1e-6,1-1e-6);d=[c[0],e[0],g[0]]
  out.append(c+e+g+[np.log(p/(1-p)),np.log(max(row.ratio,1e-3)),max(d)-min(d),np.mean(d)])
 return np.asarray(out,float)

def yandex_frame():
 labels=pd.read_csv('data/train/yandex_ocr_real_long_lines_v4/labels.csv').set_index('image_id');base=pd.read_csv('outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv').set_index('image_id');raw=pd.read_csv('outputs/yandex_real_long_v4_multicrop_raw.csv').set_index('image_id')
 p=apply_temp(.20*base.p_x025_tta.to_numpy()+.55*apply_temp(base.p_x1_tta.to_numpy(),.02)+.25*base.p_rapid.to_numpy(),.5);wide=labels.output_ratio.to_numpy(float)>=10
 for loc,image_id in zip(np.flatnonzero(wide),labels.index[wide]):
  row=raw.loc[image_id];windows=[.05*row[f'x025_{pos}']+.65*apply_temp(np.asarray([row[f'x1_{pos}']]),.02)[0]+.30*row[f'rapid_{pos}'] for pos in ('left','center','right')];p[loc]=apply_temp(np.asarray([np.mean(windows)]),.5)[0]
 return pd.DataFrame({'image_id':labels.index,'target':labels.target,'ratio':labels.output_ratio,'p_current':p}).reset_index(drop=True)

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--sample',type=Path,default=Path('data/test/sample_submission.csv'));p.add_argument('--images',type=Path,default=Path('data/test/test/images'));p.add_argument('--base',type=Path,default=Path('submission_three_model_two_regime_tuned.csv'));p.add_argument('--cyr',type=Path,default=Path('outputs/test_cyrillic_v5_rec_predictions_b8.csv'));p.add_argument('--eslav',type=Path,default=Path('outputs/test_eslav_v5_rec_predictions_b8.csv'));p.add_argument('--eng',type=Path,default=Path('outputs/test_english_v5_rec_predictions_b8.csv'));p.add_argument('--output',type=Path,default=Path('submission_ocr_v5_meta_yandex_selected_b8.csv'));return p.parse_args()

def test_frame(args,ids):
 base=pd.read_csv(args.base)
 if base.image_id.tolist()!=ids:raise ValueError('Base predictions do not match sample order')
 ratios=[]
 for image_id in ids:
  image=cv2.imread(str(args.images/f'{image_id}.png'))
  if image is None:raise RuntimeError(f'Cannot read {image_id}')
  ratios.append(image.shape[1]/image.shape[0])
 return pd.DataFrame({'image_id':ids,'ratio':ratios,'p_current':base.p_180})

def main():
 args=parse_args();random.seed(SEED);np.random.seed(SEED);ids=pd.read_csv(args.sample).image_id.tolist()
 if len(ids)!=20000 or len(set(ids))!=20000:raise ValueError('Expected 20,000 unique IDs')
 lm=CharLM(corpus());train=add_ocr(yandex_frame(),'outputs/yandex_cyrillic_v5_rec_predictions.csv','outputs/yandex_eslav_v5_rec_predictions.csv','outputs/yandex_english_v5_rec_predictions.csv');test=add_ocr(test_frame(args,ids),args.cyr,args.eslav,args.eng)
 model=make_pipeline(StandardScaler(),LogisticRegression(C=C_VALUE,max_iter=4000,class_weight='balanced',random_state=SEED));model.fit(features(train,lm)[:,FEATURE_COLUMNS],train.target.to_numpy(int));prob=apply_temp(model.predict_proba(features(test,lm)[:,FEATURE_COLUMNS])[:,1],TEMPERATURE)
 result=pd.DataFrame({'image_id':ids,'p_180':prob});result.to_csv(args.output,index=False,float_format='%.10f')
 report={'seed':SEED,'model_selection':'Yandex source-group OOF','meta_training_rows':len(train),'feature_variant':'russian_only','C':C_VALUE,'temperature':TEMPERATURE,'ocr_batch_size':8,'submission':{'path':str(args.output),'rows':len(result),'min':float(prob.min()),'max':float(prob.max()),'mean':float(prob.mean())}}
 args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
