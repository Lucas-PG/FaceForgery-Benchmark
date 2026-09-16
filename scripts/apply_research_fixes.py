"""One-time, exact-context maintenance patch; confined to this fork's work branch.

This script is used by the branch-isolated maintenance workflow. It never
invokes GitHub/network APIs. Context mismatches fail rather than applying fuzzy
edits. It is not part of the experiment execution path.
"""
from pathlib import Path


def replace(path,old,new,count=1):
    p=Path(path); s=p.read_text()
    if s.count(old)!=count: raise ValueError(f'Unexpected source context: {path}: {old[:80]}')
    p.write_text(s.replace(old,new))


def main():
    for name in ['resnet','mobilenet','dino','xception']:
        path=f'src/models/{name}.py'
        replace(path,'config.regime == "finetune"',f'uses_pretraining(config, "{name}")')
        replace(path,'def build(config) -> nn.Module:',f'from src.models._regime import uses_pretraining\n\n\ndef build(config) -> nn.Module:')
    for name in ['clip','vit']:
        path=f'src/models/{name}.py'
        replace(path,'if config.regime == "scratch":',f'if not uses_pretraining(config, "{name}"):')
        replace(path,'def build(config) -> nn.Module:',f'from src.models._regime import uses_pretraining\n\n\ndef build(config) -> nn.Module:')
    replace('src/pipelines/config.py','    regime: str = "scratch"','    regime: str = "scratch"\n    initialization_contract: str = "regime-v2"')
    replace('src/pipelines/config.py','        if self.epochs < 1',
            '        if self.initialization_contract not in {"regime-v2", "legacy-v1"}:\n            raise ValueError("Unknown initialization contract")\n        if self.epochs < 1')
    replace('src/pipelines/checkpoints.py','    raw_cfg = _read_run_config(run)',
            '    raw_cfg = _read_run_config(run)\n    # Preserve unversioned historical constructors, especially Xception/CLIP/ViT.\n    raw_cfg.setdefault("initialization_contract", "legacy-v1")')
    p=Path('src/data/data.py'); s=p.read_text()
    begin=s.index('        except Exception as e:\n',s.index('    def __getitem__'))
    end=s.index('\n        label = ',begin)
    s=s[:begin]+'        except Exception as e:\n            raise RuntimeError(f"Unreadable image at original index {idx}: {img_path}; substitution is forbidden") from e\n'+s[end:]
    p.write_text(s)
    replace('src/robustness/imaging.py','        image_hash=digest_file(path) if self.hash_images else ""\n        expected=str(row.get("sha256",""))\n        if self.hash_images and expected and expected!=image_hash:',
            '        expected=str(row.get("sha256",""))\n        image_hash=digest_file(path) if self.hash_images or expected else ""\n        if expected and expected!=image_hash:')
    replace('src/robustness/experiments.py',"    run_id=digest(identity)",
            "    if config['training']['teacher_run']:\n        from .provenance import digest_file\n        teacher = Path(config['training']['teacher_run'])\n        identity['teacher_checkpoint_sha256'] = digest_file(teacher/'best.pt')\n        identity['teacher_run_sha256'] = digest_file(teacher/'run.json')\n    run_id=digest(identity)")
    replace('src/robustness/inference.py','from .statistics import aggregate_videos',
            'from .artifacts import save_predictions\nfrom .statistics import aggregate_videos')
    replace('src/robustness/inference.py','model_sha256:str):','model_sha256:str, checkpoint_class1:str="fake"):')
    replace('src/robustness/inference.py','    if manifest_record["split"]!="val":',
            '    if checkpoint_class1 not in {"fake", "real"}: raise ValueError("Invalid score orientation")\n    if manifest_record["split"]!="val":')
    replace('src/robustness/inference.py','"selection_split":"val","label_convention":"fake-is-1",',
            '"selection_split":"val","label_convention":"fake-is-1","checkpoint_class1":checkpoint_class1,')
    replace('src/robustness/inference.py','image_size:int, mode=None, in_channels=None, positive_class="fake", **kwargs):',
            'image_size:int, mode=None, in_channels=None, positive_class="fake", research_run=None, **kwargs):')
    replace('src/robustness/inference.py',"    if calibration.get('model_sha256')!=checksum or calibration.get('selection_split')!='val':",
            "    if calibration.get('checkpoint_class1','fake')!=positive_class:\n        raise ValueError('Calibration score orientation differs from this checkpoint interpretation')\n    if calibration.get('model_sha256')!=checksum or calibration.get('selection_split')!='val':")
    replace('src/robustness/inference.py',"        write_csv(output/'predictions.csv',p)",
            "        if research_run is not None: report['research_run']=research_run\n        save_predictions(output/'predictions.csv',p,manifest_record=record,model_sha256=checksum,checkpoint_class1=positive_class)")
    replace('src/robustness/inference.py',"['predictions.csv','metrics.json']", "['predictions.csv','predictions.csv.json','metrics.json']")
    replace('src/robustness/engine.py','from .inference import predict, calibrate',
            'from .inference import predict, calibrate\nfrom .artifacts import save_predictions')
    replace('src/robustness/engine.py',"        write_csv(root/'validation_predictions.csv',validation)",
            "        save_predictions(root/'validation_predictions.csv',validation,manifest_record=val_certificate,model_sha256=digest_file(root/'best.pt'))")
    replace('src/robustness/engine.py',"files=['best.pt','last.pt','calibration.json','validation_predictions.csv','run.json','history.json']",
            "files=['best.pt','last.pt','calibration.json','validation_predictions.csv','validation_predictions.csv.json','run.json','history.json']")
    replace('src/robustness/analysis.py','from .manifests import read_csv',
            'from .manifests import read_csv\nfrom .artifacts import load_predictions')
    replace('src/robustness/analysis.py',"    a,b=align(read_csv(reference),read_csv(other))\n",'')
    replace('src/robustness/analysis.py',"    cb=json.loads(Path(other_calibration).read_text())",
            "    cb=json.loads(Path(other_calibration).read_text())\n    a=load_predictions(reference,calibration=ca)[0]\n    b=load_predictions(other,calibration=cb)[0]\n    a,b=align(a,b)")
    replace('src/robustness/analysis.py',"reference=checked_predictions(read_csv(prediction_files[0]))","reference=load_predictions(prediction_files[0])[0]")
    replace('src/robustness/analysis.py',"        _,frame=align(reference,read_csv(file))\n        record=json.loads(Path(cal).read_text())",
            "        record=json.loads(Path(cal).read_text())\n        _,frame=align(reference,load_predictions(file,calibration=record)[0])")
    replace('research_cli.py','    args=p.parse_args(argv)',
            "    from src.robustness.commands import register,run as extra_command\n    register(commands)\n    args=p.parse_args(argv)")
    replace('research_cli.py',"    if args.command=='convert-manifest':",
            "    if args.command in {'import-predictions','calibrate-export','expand-plan','profile','export-xai','report-generators','aggregate-seeds'}:\n        result=extra_command(args)\n    elif args.command=='convert-manifest':")
    replace('research_cli.py',"device=args.device,batch_size=args.batch_size,workers=args.workers)\n    elif args.command in {'evaluate-legacy'",
            "device=args.device,batch_size=args.batch_size,workers=args.workers,\n                        research_run={'name':record['config']['name'],'seed':record['config']['training']['seed'],'run_id':record['run_id']})\n    elif args.command in {'evaluate-legacy'")
    replace('research_cli.py',"            write_csv(out/'validation_predictions.csv',predictions)",
            "            from src.robustness.artifacts import save_predictions\n            save_predictions(out/'validation_predictions.csv',predictions,manifest_record=record,model_sha256=digest_file(args.checkpoint),checkpoint_class1=args.class_one)")
    replace('research_cli.py',"model_sha256=digest_file(args.checkpoint))",
            "model_sha256=digest_file(args.checkpoint),checkpoint_class1=args.class_one)")
    replace('research/robustness/STATUS.md','signed manifest certificates','integrity-certified manifests')


if __name__=='__main__': main()
