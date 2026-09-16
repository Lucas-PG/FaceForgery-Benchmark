"""Explicit inference artifacts. This module never trains or tunes on target data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .imaging import CanonicalDataset, PREPROCESSING, encode_tensor
from .manifests import load_manifest
from .provenance import SCHEMA, digest, digest_file, source_identity, write_csv, write_json
from .statistics import aggregate_videos, choose_threshold, generator_metrics, summary, checked_predictions


def predict(model, frame, root, *, image_size:int, device="cpu", batch_size=32,
            workers=0, mode:str|None=None, in_channels=None, positive_class="fake", hash_images=True):
    if positive_class not in {"fake","real"}: raise ValueError("Declare checkpoint class-1 meaning")
    if batch_size<1 or workers<0: raise ValueError("Invalid loader settings")
    device=torch.device(device)
    ds=CanonicalDataset(frame,root,image_size,hash_images=hash_images)
    loader=DataLoader(ds,batch_size=batch_size,num_workers=workers,shuffle=False)
    model=model.to(device).eval()
    rows=[]
    with torch.inference_mode():
        for batch in loader:
            raw=batch["image"].to(device)
            inputs=encode_tensor(raw,mode,in_channels) if mode is not None else raw
            logits=model(inputs)
            if logits.ndim!=2 or logits.shape!=(len(raw),2) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite two-class logits; no silent numerical replacement")
            p=logits.float().softmax(-1)[:,1 if positive_class=="fake" else 0].cpu().numpy()
            for j,position in enumerate(batch["index"].tolist()):
                row=frame.iloc[position].to_dict()
                if row["sample_id"]!=batch["sample_id"][j]: raise ValueError("Loader changed sample identity")
                row.update(p_fake=float(p[j]),image_sha256=batch["image_sha256"][j])
                rows.append(row)
    result=checked_predictions(__import__('pandas').DataFrame(rows))
    if set(result.sample_id)!=set(frame.sample_id) or len(result)!=len(frame):
        raise ValueError("Incomplete inference population")
    return result


def calibrate(predictions, *, manifest_record:dict, output:str|Path, model_sha256:str):
    if manifest_record["split"]!="val": raise ValueError("Threshold fitting requires source validation split=val")
    output=Path(output)
    if output.exists(): raise FileExistsError("Calibration is frozen; use a new artifact path")
    p=checked_predictions(predictions)
    record={"schema":SCHEMA,"model_sha256":model_sha256,"source_manifest_sha256":manifest_record['manifest_sha256'],
            "selection_split":"val","label_convention":"fake-is-1","frame_threshold":choose_threshold(p.label,p.p_fake),
            "method":"maximum validation balanced accuracy; smallest threshold tie break"}
    if "video_id" in p and p.video_id.astype(str).str.strip().ne("").all():
        videos=aggregate_videos(p)
        record['video_threshold']=choose_threshold(videos.label,videos.p_fake)
        record['video_aggregation']='mean frame fake probability'
    write_json(output,record)
    return record


def evaluation_report(predictions, calibration:dict, checkpoint_hash:str) -> dict:
    if calibration.get("schema")!=SCHEMA or calibration.get("model_sha256")!=checkpoint_hash or calibration.get("selection_split")!="val":
        raise ValueError("Calibration does not belong to this checkpoint/source-validation protocol")
    p=checked_predictions(predictions)
    result={"frame":summary(p.label,p.p_fake,float(calibration['frame_threshold'])),
            "calibration":calibration,"label_convention":"fake-is-1"}
    if "video_id" in p and p.video_id.astype(str).str.strip().ne("").all():
        videos=aggregate_videos(p)
        t=calibration.get('video_threshold',calibration['frame_threshold'])
        result['video']=summary(videos.label,videos.p_fake,float(t))
        result['video']['threshold_policy']='source-video validation' if 'video_threshold' in calibration else 'source-frame validation threshold transferred without target tuning'
        result['video']['aggregation']='mean fake probability; one observation per video'
    if {'generator','source_domain'} <= set(p):
        result['per_generator']=generator_metrics(p,float(calibration['frame_threshold']))
    return result


def evaluate(model, manifest, root, output, *, checkpoint_path, calibration_path,
             image_size:int, mode=None, in_channels=None, positive_class="fake", **kwargs):
    output=Path(output)
    if output.exists(): raise FileExistsError("Use a new evaluation output directory; stale caches are not reused")
    frame,record=load_manifest(manifest)
    checksum=digest_file(checkpoint_path)
    calibration=json.loads(Path(calibration_path).read_text())
    # Check identities before expensive work.
    if calibration.get('model_sha256')!=checksum or calibration.get('selection_split')!='val':
        raise ValueError("Unmatched or non-validation calibration")
    output.mkdir(parents=True)
    write_json(output/'status.json',{'state':'running','manifest_sha256':record['manifest_sha256']})
    try:
        p=predict(model,frame,root,image_size=image_size,mode=mode,in_channels=in_channels,
                  positive_class=positive_class,**kwargs)
        report=evaluation_report(p,calibration,checksum)
        report.update(schema=SCHEMA,manifest=record,checkpoint_sha256=checksum,
                      checkpoint_class1=positive_class,preprocessing=PREPROCESSING,software=source_identity())
        write_csv(output/'predictions.csv',p)
        if 'video' in report: write_csv(output/'video_predictions.csv',aggregate_videos(p))
        write_json(output/'metrics.json',report)
        write_json(output/'status.json',{'state':'complete','expected':len(frame),'observed':len(p),
                   'artifacts':{n:digest_file(output/n) for n in ['predictions.csv','metrics.json']}})
        return report
    except Exception as error:
        write_json(output/'status.json',{'state':'failed','error':f'{type(error).__name__}: {error}'})
        raise
