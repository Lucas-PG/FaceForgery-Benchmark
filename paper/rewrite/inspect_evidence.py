"""Read-only evidence inspection of the pinned upstream release, not training."""
from pathlib import Path
import json
import hashlib
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ['resnet','mobilenet','xception','vit','clip','dino']
SEEDS = [42,123,2024,7,2025]
pd.set_option('display.max_columns',50)
pd.set_option('display.width',250)


def show(name, frame):
    print('\n=== '+name+' ===\n'+frame.to_string(index=False))


def main():
    raw = pd.read_csv(ROOT/'tables/results_seed_level.csv')
    print('BASELINE SCHEMA', raw.columns.tolist(), 'REGIMES', raw.regime.value_counts().to_dict())
    print('BASELINE DUPLICATES', int(raw.duplicated(['model_family','fourier_mode','regime','seed']).sum()))
    base = raw.loc[raw.regime.eq('finetune') & raw.fourier_mode.eq('none') & raw.seed.isin(SEEDS)].copy()
    robust = pd.concat([pd.read_csv(ROOT/f'tables/robust_5seeds_gpu{i}.csv') for i in (0,1)],ignore_index=True)
    show('BASELINE RGB INDIVIDUAL',base[['model_family','seed','test_auc','test_d_auc']])
    print('ROBUST COUNTS',robust.groupby('model_family').seed.agg(list).to_dict())
    records = []
    for family in FAMILIES:
        a=base.loc[base.model_family.eq(family)].set_index('seed').loc[SEEDS]
        b=robust.loc[robust.model_family.eq(family)].set_index('seed').loc[SEEDS]
        assert len(a)==len(b)==5
        for metric in ['test_auc','test_d_auc']:
            delta=b[metric]-a[metric]
            records.append(dict(family=family,metric=metric,baseline_mean=a[metric].mean(),baseline_sd=a[metric].std(),robust_mean=b[metric].mean(),robust_sd=b[metric].std(),paired_gain=delta.mean(),paired_gain_sd=delta.std(),gains_positive=int((delta>0).sum())))
    show('PRIMARY COMMON FIVE-SEED COMPARISON',pd.DataFrame(records))
    df40=pd.read_csv(ROOT/'tables/df40_all_runs_individual.csv')
    print('DF40 INDIVIDUAL REGIMES',df40.regime.value_counts().to_dict())
    d=df40.loc[df40.regime.eq('finetune') & df40.fourier_mode.eq('none') & df40.seed.isin(SEEDS)]
    show('BASELINE DF40 COMMON FIVE',d[['model_family','seed','auc']])
    show('ROBUST DF40 COMMON FIVE',robust[['model_family','seed','df40_auc']])
    rows=[]
    for family in FAMILIES:
        a=d.loc[d.model_family.eq(family)].set_index('seed').loc[SEEDS].auc
        b=robust.loc[robust.model_family.eq(family)].set_index('seed').loc[SEEDS].df40_auc
        rows.append(dict(family=family,base_mean=a.mean(),base_sd=a.std(),robust_mean=b.mean(),robust_sd=b.std(),paired_gain=(b-a).mean(),paired_gain_sd=(b-a).std(),positive=int((b-a>0).sum())))
    show('DF40 PAIRED COMPARISON',pd.DataFrame(rows))
    spectral=raw.loc[raw.regime.eq('finetune') & raw.seed.isin(SEEDS)].groupby(['model_family','fourier_mode']).agg(n=('seed','count'),clean=('test_auc','mean'),degraded=('test_d_auc','mean')).reset_index()
    show('ALL SPATIAL SPECTRAL MODES FIVE SEEDS',spectral)
    ens=pd.read_csv(ROOT/'tables/ensemble_robust_canonical_5seeds.csv')
    print('ENSEMBLE TYPES',ens.type.value_counts().to_dict())
    show('ENSEMBLE: CLIP DINO, ALL SIX, SELF ENSEMBLES AND GLOBAL',ens.loc[(ens.id.eq('clip_dino')) | ens.k.ge(6) | ens.type.ne('multi_model'),['type','k','id','label','members','strategy','n_seeds','val_auc_m','val_auc_s','test_auc_m','test_auc_s','test_d_auc_m','test_d_auc_s','df40_auc_m','df40_auc_s']])
    print('\nTRAIN VAL TEST COUNTS AND ID CHECK')
    manifests={}
    for split in ['train','val','test']:
        p=ROOT/f'data/raw/{split}.csv'; f=pd.read_csv(p)
        manifests[split]=f
        print(split,'rows',len(f),'columns',f.columns.tolist(),'labels',f.iloc[:,1].value_counts().to_dict(),'duplicate_image_names',int(f.iloc[:,0].duplicated().sum()),'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
    for a,b in [('train','val'),('train','test'),('val','test')]:
        print('EXACT FILENAME OVERLAP',a,b,len(set(manifests[a].iloc[:,0]) & set(manifests[b].iloc[:,0])))
    print('\nROOT ENSEMBLE OUTPUTS')
    print((ROOT/'ensemble_report.csv').read_text())
    for split in ['val','test','test_d']:
        p=ROOT/f'ensemble_predictions_{split}.csv'; f=pd.read_csv(p)
        print(split,len(f),f.columns.tolist(),f.head(2).to_dict('records'))
        if {'y_true','prob_pos'} <= set(f): print('AUC',roc_auc_score(f.y_true,f.prob_pos),'labels',f.y_true.value_counts().to_dict(),'duplicate_ids',int(f.id.duplicated().sum()) if 'id' in f else None)
    print('\nEXACT IMPORTANT CONFIG/SCRIPTS')
    for name in ['configs/base.yaml','configs/dino.yaml','configs/xception.yaml','src/data/augmentations.py','scripts/evaluate_celeb_df_robust_ensembles.py','results/xai_heatmaps/README.md','scripts/generate_xai_heatmaps.py']:
        print('\n=== '+name+' ===\n'+(ROOT/name).read_text())
    print('\nTRAINING / ENSEMBLE PROVENANCE LINES')
    for name in ['scripts/train_robust_seeds.py','scripts/_robust_common.py','scripts/build_tabela7_robust_ensembles.py','scripts/build_all_rayson_tabelas.py','src/pipelines/training.py','scripts/prepare_df40_benchmark.py']:
        lines=(ROOT/name).read_text().splitlines()
        print('\nFILE',name)
        selected=set()
        for i,line in enumerate(lines):
            if any(s in line.lower() for s in ['seed','epoch','regime','image_size','pretrain','optimizer','lr_head','lr_backbone','batch_size','geometric','logistic','ddof','canonical','cdf/frames','val_auc','sample_size','normalize','choice']):
                selected.update(range(max(0,i-1),min(len(lines),i+3)))
        for i in sorted(selected): print(f'{i+1}: {lines[i]}')
    print('\nCROSS-DATASET COUNTS / CORRECTION NOTES')
    for name in ['results/mostrar_rayson/tabela5-crossdata-df40.md','results/mostrar_rayson/tabela6-crossdata-celebdf.md','results/mostrar_rayson/tabela7-ensemble-robusto.md']:
        lines=(ROOT/name).read_text().splitlines()
        for i,line in enumerate(lines):
            if any(t in line.lower() for t in ['corrig','polar','0.9614','0.8845','0.8610','0.654','0.658','518','13.000','11.000','10951','10.951','10320','10.320','metodologia','extrai','27.72','72.60']): print(name, i+1, line)


if __name__=='__main__': main()
