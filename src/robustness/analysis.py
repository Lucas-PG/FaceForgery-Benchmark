"""Artifact-only analysis: no GPU, classifier loading, or inferred experiment results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .manifests import read_csv
from .artifacts import load_predictions
from .provenance import digest_file, write_csv, write_json
from .statistics import align, checked_predictions, complementarity, grouped_auc_interval, summary


def compare_files(reference,other,output,reference_calibration,other_calibration,*,draws=1000,seed=42):
    ca=json.loads(Path(reference_calibration).read_text())
    cb=json.loads(Path(other_calibration).read_text())
    a=load_predictions(reference,calibration=ca)[0]
    b=load_predictions(other,calibration=cb)[0]
    a,b=align(a,b)
    if ca.get('selection_split')!='val' or cb.get('selection_split')!='val':
        raise ValueError('Comparison requires validation-frozen thresholds')
    result,cases=complementarity(a,b,reference_threshold=ca['frame_threshold'],other_threshold=cb['frame_threshold'])
    result['paired_auc_interval']=grouped_auc_interval(a,b,draws=draws,seed=seed)
    result['reference']=summary(a.label,a.p_fake,ca['frame_threshold'])
    result['other']=summary(b.label,b.p_fake,cb['frame_threshold'])
    result['input_sha256']={k:digest_file(v) for k,v in [('reference_predictions',reference),('other_predictions',other),
                                                        ('reference_calibration',reference_calibration),('other_calibration',other_calibration)]}
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a fresh comparison directory')
    output.mkdir(parents=True)
    write_json(output/'comparison.json',result)
    write_csv(output/'paired_cases.csv',cases)
    return result


def shared_failures(prediction_files,calibrations,output):
    if len(prediction_files)<2 or len(prediction_files)!=len(calibrations):
        raise ValueError('Declare at least two matched predictions/calibrations')
    reference=load_predictions(prediction_files[0])[0]
    failed=reference.label.eq(reference.label).to_numpy()
    roster=[]
    for file,cal in zip(prediction_files,calibrations):
        record=json.loads(Path(cal).read_text())
        _,frame=align(reference,load_predictions(file,calibration=record)[0])
        if record.get('selection_split')!='val': raise ValueError('Threshold must be selected on validation')
        t=record['frame_threshold']; summary(frame.label,frame.p_fake,t)
        failed &= (frame.p_fake.to_numpy()>=t) != frame.label.to_numpy()
        roster.append({'predictions_sha256':digest_file(file),'calibration_sha256':digest_file(cal)})
    if len({r['predictions_sha256'] for r in roster})!=len(roster): raise ValueError('Duplicate model prediction artifacts in roster')
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a new shared-failure output directory')
    output.mkdir(parents=True)
    write_csv(output/'shared_failures.csv',reference.loc[failed])
    result={'n_population':len(reference),'n_shared_failures':int(failed.sum()),'rate':float(failed.mean()),
            'roster':roster,'scope':'these exact checkpoints and thresholds, not universal failure'}
    write_json(output/'summary.json',result)
    return result


def inventory_runs(root,output,required_splits=('val','test','test_d')):
    """Never equate weights/best.pth with a completed training/evaluation run."""
    root=Path(root); rows=[]
    for checkpoint in sorted(root.glob('*/*/*/seed_*/weights/best.pth')):
        run=checkpoint.parent.parent; results=run/'results'
        family,mode,regime,seed=checkpoint.relative_to(root).parts[:4]
        errors=[]; counts={}
        for split in required_splits:
            path=results/f'predictions_{split}.csv'
            if not path.exists(): errors.append(f'missing predictions_{split}.csv'); continue
            frame=pd.read_csv(path)
            if not {'id','y_true','prob_pos'}<=set(frame): errors.append(f'invalid schema: {split}'); continue
            try:
                canonical=pd.DataFrame({'sample_id':frame.id.astype(str),'group_id':frame.id.astype(str),
                                        'label':frame.y_true,'p_fake':frame.prob_pos})
                checked_predictions(canonical)
                counts[split]=len(frame)
            except ValueError as error: errors.append(f'{split}: {error}')
        cfg=results/'run_config.json'
        if not cfg.exists(): errors.append('missing run configuration')
        rows.append({'family':family,'mode':mode,'regime':regime,'seed':seed,
                     'checkpoint_sha256':digest_file(checkpoint),'prediction_rows':counts,'issues':errors,
                     'state':'incomplete_or_invalid' if errors else 'artifacts_present_not_reproduced',
                     'warning':'Historical initialization/label/provenance not proved by path names; not certified complete.'})
    result={'runs':rows,'n_runs':len(rows),'required_splits':list(required_splits),
            'note':'No metric correction or missing-seed imputation has been applied.'}
    write_json(output,result)
    return result
