"""Generate the paper's evidence, tables and figure; no model execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/six-page/generated'
PIN = 'd156d77897288816f6b619d577cc77d374ea0013'
SEEDS = [7, 42, 123, 2024, 2025]
FAMILIES = ['resnet','mobilenet','xception','vit','clip','dino']
NAMES = {'resnet':'ResNet-18','mobilenet':'MobileNetV3-L','xception':'Xception','vit':'ViT-B/16','clip':'CLIP ViT-B/16','dino':'DINOv3 ConvNeXt-B'}
SHORT = {'resnet':'ResNet','mobilenet':'MobileNet','xception':'Xception','vit':'ViT','clip':'CLIP','dino':'DINOv3'}
MODES = ['none','magnitude','phase','complex','frequency_3','concat','concat_frequency']
SOURCES = ['tables/results_seed_level.csv','tables/robust_5seeds_gpu0.csv','tables/robust_5seeds_gpu1.csv','tables/df40_all_runs_individual.csv','tables/ensemble_robust_canonical_5seeds.csv','tables/celeb_df_robust_ensembles.csv','data/raw/train.csv','data/raw/val.csv','data/raw/test.csv']
B = chr(92)
NL = chr(10)
ROW = ' ' + B*2


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+NL)


def load_source(name, ledger):
    path = ROOT / name
    expected = subprocess.check_output(['git','rev-parse',f'{PIN}:{name}'],cwd=ROOT,text=True).strip()
    observed = subprocess.check_output(['git','hash-object',str(path)],cwd=ROOT,text=True).strip()
    if expected != observed:
        raise ValueError(f'Pinned evidence changed: {name}')
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    ledger[name] = {'git_blob':observed,'sha256':sha256(path),'rows':len(frame),'columns':frame.columns.tolist()}
    return frame


def cell(values):
    x = np.asarray(values,dtype=float)*100
    return f'${x.mean():.2f} {B}pm {x.std(ddof=1):.2f}$'


def delta_cell(values):
    x = np.asarray(values,dtype=float)*100
    return f'${x.mean():+.2f} {B}pm {x.std(ddof=1):.2f}$'


def table(path,spec,header,rows):
    text = B+'begin{tabular}{'+spec+'}'+NL+B+'toprule'+NL+header+NL+B+'midrule'+NL
    text += NL.join(rows)+NL+B+'bottomrule'+NL+B+'end{tabular}'+NL
    (OUT/path).write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ledger = {}
    data = {name:load_source(name,ledger) for name in SOURCES}
    raw = data[SOURCES[0]]
    if len(raw)!=210 or raw.duplicated(['model_family','fourier_mode','regime','seed']).any() or set(raw.regime)!={'finetune'}:
        raise ValueError('Standard records are not the unique 210-run fine-tuned matrix')
    for family in FAMILIES:
        for mode in MODES:
            block = raw.loc[raw.model_family.eq(family)&raw.fourier_mode.eq(mode)]
            if len(block)!=5 or set(block.seed)!=set(SEEDS):
                raise ValueError(f'Incomplete standard condition: {family}/{mode}')
    robust = pd.concat([data[SOURCES[1]],data[SOURCES[2]]],ignore_index=True)
    if len(robust)!=30 or robust.duplicated(['model_family','seed']).any() or set(robust.regime)!={'finetune_robust'}:
        raise ValueError('Robust records are not 30 distinct family/seed runs')
    base = raw.loc[raw.fourier_mode.eq('none')].copy()
    external = data[SOURCES[3]]
    external = external.loc[external.regime.eq('finetune')&external.fourier_mode.eq('none')&external.seed.isin(SEEDS)]
    if len(external)!=30 or external.duplicated(['model_family','seed']).any():
        raise ValueError('External standard records are incomplete or duplicated')
    base = base.merge(external[['model_family','seed','auc']].rename(columns={'auc':'df40_auc'}),on=['model_family','seed'],validate='one_to_one')
    metrics = ['test_auc','test_d_auc','df40_auc']
    paired = base[['model_family','seed']+metrics].merge(robust[['model_family','seed']+metrics],on=['model_family','seed'],validate='one_to_one',suffixes=('_standard','_robust'))
    if len(paired)!=30:
        raise ValueError('Pairing lost records')
    for metric in metrics:
        values = paired[[metric+'_standard',metric+'_robust']].to_numpy()
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError('Invalid AUC values')
        paired[metric+'_gain'] = paired[metric+'_robust']-paired[metric+'_standard']
    paired.sort_values(['model_family','seed']).to_csv(OUT/'paired_seed_evidence.csv',index=False,float_format='%.12g')
    raw.sort_values(['model_family','fourier_mode','seed']).to_csv(OUT/'representation_seed_evidence.csv',index=False,float_format='%.12g')
    summary = []
    for family in FAMILIES:
        block = paired.loc[paired.model_family.eq(family)].set_index('seed').loc[SEEDS]
        for metric in metrics:
            a,b,d = [block[metric+'_'+suffix].to_numpy() for suffix in ['standard','robust','gain']]
            summary.append({'family':family,'metric':metric,'n_seeds':5,'standard_mean':float(a.mean()),'standard_sd':float(a.std(ddof=1)),'robust_mean':float(b.mean()),'robust_sd':float(b.std(ddof=1)),'gain_mean':float(d.mean()),'gain_sd':float(d.std(ddof=1)),'positive_pairs':int((d>0).sum())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT/'paired_summary.csv',index=False,float_format='%.12g')
    rows = []
    for family in FAMILIES:
        block = paired.loc[paired.model_family.eq(family)]
        cells = [NAMES[family],cell(block.test_auc_standard),cell(block.test_auc_robust),cell(block.test_d_auc_standard),cell(block.test_d_auc_robust),delta_cell(block.test_d_auc_gain)]
        rows.append(' & '.join(cells)+ROW)
    header = 'Model & '+B+'multicolumn{2}{c}{Clean test AUC ('+B+'%)} & '+B+'multicolumn{2}{c}{Test-D AUC ('+B+'%)} & Test-D gain (pp)'+ROW+NL
    header += B+'cmidrule(lr){2-3}'+B+'cmidrule(lr){4-5}'+NL+' & Standard & Robust & Standard & Robust & Paired difference'+ROW
    table('primary_table.tex','lccccc',header,rows)
    rows = []
    for family in FAMILIES:
        block = paired.loc[paired.model_family.eq(family)]
        cells = [SHORT[family],cell(block.df40_auc_standard),cell(block.df40_auc_robust),f'${100*block.df40_auc_gain.mean():+.2f}$',f'{int((block.df40_auc_gain>0).sum())}/5']
        rows.append(' & '.join(cells)+ROW)
    header = 'Model & Standard & Robust & Gain & Positive'+ROW+NL+' & '+B+'multicolumn{2}{c}{AUC ('+B+'%), mean $'+B+'pm$ SD} & (pp) & pairs'+ROW
    table('transfer_table.tex','lcccc',header,rows)
    rows,representations = [],[]
    for family in FAMILIES:
        cells = [SHORT[family]]
        for mode in MODES:
            block = raw.loc[raw.model_family.eq(family)&raw.fourier_mode.eq(mode)]
            representations.append({'family':family,'mode':mode,'clean_mean':float(block.test_auc.mean()),'degraded_mean':float(block.test_d_auc.mean()),'n_seeds':len(block)})
            cells.append(f'{100*block.test_auc.mean():.2f} / {100*block.test_d_auc.mean():.2f}')
        rows.append(' & '.join(cells)+ROW)
    table('representation_table.tex','lccccccc','Model & RGB & Magnitude & Phase & Complex & High-pass & RGB+M & Hybrid'+ROW,rows)
    write_json(OUT/'representation_summary.json',representations)
    manifest_rows,manifests = [],{}
    for split in ['train','val','test']:
        frame = data[f'data/raw/{split}.csv']
        if set(frame.target)!={0,1} or frame.img_name.duplicated().any():
            raise ValueError('Manifest class or filename identity failure')
        manifests[split]=frame
        counts=frame.target.value_counts()
        label={'train':'Train','val':'Validation','test':'Clean test'}[split]
        manifest_rows.append(f'{label} & {counts[0]:,} & {counts[1]:,} & {len(frame):,}'+ROW)
    overlap={a+'-'+b:len(set(manifests[a].img_name)&set(manifests[b].img_name)) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlap.values()):
        raise ValueError('Unexpected cross-split filename overlap')
    table('population_table.tex','lrrr','Partition & Real & Fake & Total'+ROW,manifest_rows)
    ensembles=data[SOURCES[4]]
    print('ENSEMBLE STRATEGIES',ensembles.strategy.unique().tolist())
    geo_names=[name for name in ensembles.strategy.unique() if 'geom' in name.lower()]
    if len(geo_names)!=1:
        raise ValueError('Ambiguous geometric fusion label')
    geo=geo_names[0]
    selected=ensembles.loc[ensembles.type.eq('multi_model')&ensembles.id.isin(['clip_dino','all_6_models'])&ensembles.strategy.isin(['mean',geo,'stacking'])].copy()
    if len(selected)!=6 or selected.duplicated(['id','strategy']).any() or not selected.n_seeds.eq(5).all():
        raise ValueError('Missing or ambiguous ensemble summaries')
    rows=[]
    for ident,label in [('clip_dino','CLIP+DINOv3'),('all_6_models','All six')]:
        for strategy in ['mean',geo,'stacking']:
            row=selected.loc[selected.id.eq(ident)&selected.strategy.eq(strategy)].iloc[0]
            cells=[label,{'mean':'Mean','stacking':'Stacking'}.get(strategy,'Geometric')]
            for metric in ['val_auc','test_auc','test_d_auc','df40_auc']:
                cells.append(f'${100*row[metric+"_m"]:.2f} {B}pm {100*row[metric+"_s"]:.2f}$')
            rows.append(' & '.join(cells)+ROW)
    table('ensemble_table.tex','llcccc','Members & Fusion & Validation & Clean test & Test-D & DF40 subset'+ROW,rows)
    selected.to_csv(OUT/'ensemble_summary_evidence.csv',index=False)
    dmeans=summary_df.loc[summary_df.metric.eq('test_d_auc'),'gain_mean']*100
    pure=pd.DataFrame(representations).query("mode in ['magnitude','phase','complex','frequency_3']")
    numbers={'MinDegradationGain':f'{dmeans.min():.2f}','MaxDegradationGain':f'{dmeans.max():.2f}','PositiveDegradationPairs':str(int((paired.test_d_auc_gain>0).sum())),'PureMinimum':f'{100*pure.degraded_mean.min():.2f}','PureMaximum':f'{100*pure.degraded_mean.max():.2f}'}
    for family in FAMILIES:
        prefix={'resnet':'Resnet','mobilenet':'Mobile','xception':'Xception','vit':'Vit','clip':'Clip','dino':'Dino'}[family]
        block=paired.loc[paired.model_family.eq(family)]
        for metric,m in [('test_auc','Clean'),('test_d_auc','Degraded'),('df40_auc','External')]:
            for suffix,s in [('standard','Standard'),('robust','Robust'),('gain','Gain')]:
                numbers[prefix+m+s]=f'{100*block[metric+"_"+suffix].mean():.2f}'
        numbers[prefix+'ExternalGainAbs']=f'{abs(100*block.df40_auc_gain.mean()):.2f}'
        numbers[prefix+'CleanGainAbs']=f'{abs(100*block.test_auc_gain.mean()):.2f}'
    (OUT/'numbers.tex').write_text('% Generated from pinned per-seed evidence; AUC percent / gain percentage points.'+NL+NL.join(B+'newcommand{'+B+key+'}{'+value+'}' for key,value in numbers.items())+NL)
    fig,ax=plt.subplots(figsize=(3.48,2.70))
    pos=np.arange(len(FAMILIES))
    for metric,label,offset,marker in [('test_d_auc','Test-D',-.12,'o'),('df40_auc','DF40 subset',.12,'s')]:
        block=summary_df.loc[summary_df.metric.eq(metric)].set_index('family').loc[FAMILIES]
        ax.errorbar(block.gain_mean.to_numpy()*100,pos+offset,xerr=block.gain_sd.to_numpy()*100,fmt=marker,markersize=3.4,capsize=2,elinewidth=.8,label=label)
    ax.axvline(0,linewidth=.7,linestyle=':')
    ax.set_yticks(pos,[SHORT[f] for f in FAMILIES],fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Robust minus standard AUC (percentage points)',fontsize=8)
    ax.tick_params(axis='x',labelsize=8)
    ax.legend(fontsize=7.7,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.6)
    fig.savefig(OUT/'augmentation_gains.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/'augmentation_gains.png',dpi=220)
    plt.close(fig)
    write_json(OUT/'claims.json',{'primary_seed_set':SEEDS,'standard_runs':210,'robust_rgb_runs':30,'degradation_positive_pairs':int((paired.test_d_auc_gain>0).sum()),'source_filename_overlaps':overlap,'summary':summary,'ensemble_pool_warning':'Legacy ensemble summaries are separate from the primary 2025 replacement cohort; no pooling across these experiments.','uncertainty':'Primary SD uses ddof=1 across five seeds. No population CI or hypothesis test is inferred from aggregate AUCs.','execution':'Released-record reanalysis only; no detector training or full-dataset inference performed by this build.'})
    write_json(OUT/'source_ledger.json',{'upstream_commit':PIN,'sources':ledger,'unit':'AUC CSVs use fractions; TeX uses percent and gains use percentage points','precision':'Robust per-seed CSVs are rounded to four decimal places; aggregation cannot restore omitted precision.'})
    print('PRIMARY SUMMARY',summary_df.to_string(index=False),sep=NL)
    print('GENERATED FILES',sorted(path.name for path in OUT.iterdir()))
    print('CLAIM MACROS',json.dumps(numbers,indent=2))

if __name__=='__main__':
    main()
