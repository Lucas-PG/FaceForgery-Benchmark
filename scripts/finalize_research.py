"""One-time exact-context refinements, then removed by the guarded workflow."""
from pathlib import Path


def replace(path, old, new, count=1):
    file = Path(path)
    content = file.read_text()
    if content.count(old) != count:
        raise ValueError(f'Unexpected source context: {path}: {old[:80]}')
    file.write_text(content.replace(old, new))


def main():
    replace('src/robustness/inference.py', "            raw=batch['image'].to(device)\n            inputs=encode_legacy_tensor(raw,mode,in_channels) if mode is not None else raw",
        "            raw=batch['image']\n            inputs=encode_legacy_tensor(raw,mode,in_channels).to(device) if mode is not None else raw.to(device)")
    replace('src/robustness/engine.py', "record={**frozen,'initial_state_sha256':state_digest(model),'parameter_count':sum(p.numel() for p in model.parameters()),",
        "record={**frozen,'initial_state_sha256':state_digest(model),\n                'initial_backbone_sha256':state_digest(model.backbone) if model.backbone is not None else None,\n                'parameter_count':sum(p.numel() for p in model.parameters()),")
    replace('research_cli.py', "        from src.robustness.engine import load_pilot", "        from src.robustness.engine import load_pilot\n        from src.robustness.identity import evaluation_identity")
    replace('research_cli.py', "research_run={'name':record['config']['name'],'seed':record['config']['training']['seed'],'run_id':record['run_id']}",
        "research_run=evaluation_identity(record)")
    replace('src/robustness/commands.py', "        if not identity: raise ValueError('Seed aggregation requires pilot run metadata; do not infer seed/augmentation from filenames')",
        "        if not identity or 'condition_sha256' not in identity:\n            raise ValueError('Seed aggregation requires complete pilot condition metadata; do not infer settings from filenames')\n        from .provenance import digest\n        if digest(identity['condition']) != identity['condition_sha256']:\n            raise ValueError('Experimental-condition metadata changed')")
    replace('src/robustness/commands.py', "'seed':identity['seed'],'auc':report['frame']['auc'],'model_sha256':report['checkpoint_sha256'],",
        "'seed':identity['seed'],'auc':report['frame']['auc'],'model_sha256':report['checkpoint_sha256'],\n                     'condition_sha256':identity['condition_sha256'],")
    replace('src/robustness/commands.py', "        if group.manifest_sha256.nunique()!=1: raise ValueError('Seed runs evaluated on different populations')",
        "        if group.manifest_sha256.nunique()!=1: raise ValueError('Seed runs evaluated on different populations')\n        if group.condition_sha256.nunique()!=1: raise ValueError('Seed runs differ in model, training, source data or software conditions')")
    replace('src/robustness/explain.py', "    indices=[i for i,row in frame.iterrows() if row.sample_id in set(cohort.sample_id) and int(digest(row.sample_id)[:16],16)%args.shard_count==args.shard_index]",
        "    positions = {ident: i for i, ident in enumerate(frame.sample_id)}\n    indices = [positions[ident] for ident in cohort.sample_id\n               if int(digest(ident)[:16],16)%args.shard_count==args.shard_index]")
    replace('src/robustness/explain.py', "              'predictions_sha256':pc['predictions_sha256'],'cohort_sha256':cc['cohort_sha256'],",
        "              'predictions_sha256':pc['predictions_sha256'],'cohort_sha256':cc['cohort_sha256'],\n              'calibration_sha256':digest_file(root/'calibration.json'),")
    replace('src/robustness/explain.py', "            if abs(actual-float(by_id.loc[ident,'p_fake']))>1e-5:",
        "            if not np.isfinite(actual) or abs(actual-float(by_id.loc[ident,'p_fake']))>1e-5:")
    replace('src/robustness/explain.py', "            meta.update(contract_id=contract_id,sample_id=ident,label=item['label'],p_fake=actual,",
        "            reference_row=cohort.set_index('sample_id').loc[ident]\n            meta.update(contract_id=contract_id,sample_id=ident,label=item['label'],p_fake=actual,\n                        model_prediction=int(actual>=calibration['frame_threshold']),\n                        model_threshold=calibration['frame_threshold'],\n                        reference_stratum=str(reference_row.get('reference_stratum','not-stratified')),")
    # Enforce model bounds during planning, not after a potentially large download.
    replace('src/robustness/experiments.py', "    t,m=cfg['training'],cfg['model']",
        "    t,m=cfg['training'],cfg['model']\n    if not isinstance(m['width'],int) or isinstance(m['width'],bool) or m['width']<4 or m['width']%4:\n        raise ValueError('Auxiliary width must be a positive multiple of four')\n    if not 0 <= float(m['branch_dropout']) < 1:\n        raise ValueError('branch_dropout must be in [0,1)')")
    p = Path('docs/robustness.md')
    p.write_text(p.read_text()+'''\n## New-pilot XAI and cross-seed equivalence\n\nFor new pilot checkpoint explanations, use the separate `python -m src.robustness.explain` entry point and [pilot XAI guide](pilot-xai.md). It supports fixed-cohort IG, occlusion and declared CNN-layer Grad-CAM with fresh-score checks, hashes and verified resume. Its maps are end-to-end RGB-input attributions through the FFT path, **not native-frequency maps**. The original twelve-method native-coordinate workflow remains available separately.\n\nNew evaluation records include the model/training condition, source manifest hashes, preprocessing, augmentation definition and software identity. `aggregate-seeds` rejects runs with different conditions even if they share a display name. The seed is the excluded experimental factor; local output/data paths are not treated as scientific conditions. Teacher bytes may differ across seeds, so a consistent teacher-selection policy must still be reviewed. The initial backbone-state hash is recorded separately to verify paired initialization.\n''')
    p = Path('START_HERE.md')
    p.write_text(p.read_text()+'''\n## New-pilot explanation entry point\n\n`src/robustness/explain.py` and [docs/pilot-xai.md](docs/pilot-xai.md) connect the new pilot models to a frozen shared cohort and primary IG/occlusion/CNN Grad-CAM analyses. The raw-input maps, per-model decisions, numerical residuals and resume checks remain distinct from the preserved legacy native-frequency pipeline.\n''')


if __name__ == '__main__': main()
