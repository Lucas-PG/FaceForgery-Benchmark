"""Verify the final paper without fitting or evaluating any detector.

The numerical check independently uses csv/statistics rather than the table
builder's pandas aggregation. PDF checks supplement, not replace, visual review.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from collections import Counter

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/six-page'
PREVIEW = ROOT / 'paper/rewrite/preview'
PIN = 'd156d77897288816f6b619d577cc77d374ea0013'
SEEDS = {7,42,123,2024,2025}
B = chr(92)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def numerical_check():
    base = records(ROOT/'tables/results_seed_level.csv')
    robust = records(ROOT/'tables/robust_5seeds_gpu0.csv') + records(ROOT/'tables/robust_5seeds_gpu1.csv')
    external = records(ROOT/'tables/df40_all_runs_individual.csv')
    a = {(r['model_family'],int(r['seed'])):r for r in base if r['fourier_mode']=='none' and int(r['seed']) in SEEDS}
    b = {(r['model_family'],int(r['seed'])):r for r in robust if int(r['seed']) in SEEDS}
    c = {(r['model_family'],int(r['seed'])):r for r in external if r['regime']=='finetune' and r['fourier_mode']=='none' and int(r['seed']) in SEEDS}
    if set(a)!=set(b) or set(a)!=set(c) or len(a)!=30:
        raise ValueError('Independent seed-key reconciliation failed')
    verified = 0
    for row in records(PAPER/'generated/paired_summary.csv'):
        family,metric = row['family'],row['metric']
        old = [float(c[(family,s)]['auc'] if metric=='df40_auc' else a[(family,s)][metric]) for s in sorted(SEEDS)]
        new = [float(b[(family,s)][metric]) for s in sorted(SEEDS)]
        delta = [y-x for x,y in zip(old,new)]
        truth = {'standard_mean':statistics.mean(old),'standard_sd':statistics.stdev(old),
                 'robust_mean':statistics.mean(new),'robust_sd':statistics.stdev(new),
                 'gain_mean':statistics.mean(delta),'gain_sd':statistics.stdev(delta),
                 'positive_pairs':sum(d>0 for d in delta)}
        for key,expected in truth.items():
            if not math.isclose(float(row[key]),expected,rel_tol=1e-9,abs_tol=1e-11):
                raise ValueError(f'Independent aggregation mismatch: {family}/{metric}/{key}')
            verified += 1
    positive = sum(float(b[k]['test_d_auc'])>float(a[k]['test_d_auc']) for k in a)
    if positive != 30:
        raise ValueError('The all-30 positive Test-D claim is not supported')
    return {'independent_scalar_checks':verified,'primary_pairs':len(a),'positive_test_d_pairs':positive,
            'method':'Python csv + statistics, independent of the pandas table-generation path'}


def main():
    PREVIEW.mkdir(parents=True,exist_ok=True)
    tex = (PAPER/'main.tex').read_text()
    bib = (PAPER/'references.bib').read_text()
    log = (PAPER/'main.log').read_text(errors='replace')
    failures = []
    required = B+'documentclass[conference,10pt,letterpaper]{IEEEtran}'
    if required not in tex:
        failures.append('The required unmodified IEEEtran conference class is absent')
    for forbidden in ['usepackage{geometry}','textheight','textwidth','baselinestretch','linespread','vspace{-','scalebox']:
        if forbidden in tex:
            failures.append('Unexpected layout manipulation: '+forbidden)
    cited = set()
    for group in re.findall(r'\\cite\{([^}]+)\}',tex):
        cited.update(group.split(','))
    bibkeys = set(re.findall(r'@\w+\{([^,]+),',bib))
    if cited-bibkeys:
        failures.append('Missing bibliography entries: '+str(sorted(cited-bibkeys)))
    if bibkeys-cited:
        failures.append('Uncited bibliography entries: '+str(sorted(bibkeys-cited)))
    unresolved = [line for line in log.splitlines() if any(token in line for token in ['undefined','Rerun to get cross-references right','There were undefined','multiply defined'])]
    overfull = [line for line in log.splitlines() if 'Overfull' in line]
    if unresolved:
        failures.append('Unresolved LaTeX references/citations')
    if overfull:
        failures.append('Overfull boxes remain')
    doc = fitz.open(PAPER/'main.pdf')
    if not 6 <= len(doc) <= 8:
        failures.append(f'Page count {len(doc)} is outside 6-8 inclusive')
    pages, fonts, text_parts, thumbs = [],Counter(),[],[]
    for number,page in enumerate(doc,1):
        if abs(page.rect.width-612)>.1 or abs(page.rect.height-792)>.1:
            failures.append(f'Page {number} is not US Letter')
        words = page.get_text('words')
        text = page.get_text()
        text_parts.append(text)
        if len(words)<100:
            failures.append(f'Page {number} has fewer than 100 words')
        if '[?]' in text or chr(0xfffd) in text:
            failures.append(f'Unresolved or undecodable text on page {number}')
        bounds = [min(w[0] for w in words),min(w[1] for w in words),max(w[2] for w in words),max(w[3] for w in words)]
        outside = [w for w in words if w[0]<43 or w[2]>569 or w[1]<42 or w[3]>750]
        if outside:
            failures.append(f'Page {number} contains {len(outside)} words outside expected IEEE text bounds')
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line.get('spans',[]):
                    fonts[(span['font'],round(span['size'],2))] += len(span['text'])
        pages.append({'page':number,'words':len(words),'bounds_pt':bounds,
                      'left_column_words':sum(w[2]<306 for w in words),
                      'right_column_words':sum(w[0]>306 for w in words),
                      'first_text':text[:180],'last_text':text[-180:],
                      'out_of_bounds':[{'text':w[4],'box':list(w[:4])} for w in outside[:20]]})
        pix = page.get_pixmap(matrix=fitz.Matrix(1.65,1.65),alpha=False)
        path=PREVIEW/f'page-{number:02}.png'
        pix.save(path)
        im=Image.open(path).convert('RGB')
        im.thumbnail((306,396))
        thumb=Image.new('RGB',(326,430),'white')
        thumb.paste(im,((326-im.width)//2,12))
        ImageDraw.Draw(thumb).text((14,411),f'Page {number}',fill='black')
        thumbs.append(thumb)
    if not any(9.8 <= size <= 10.1 and count>500 for (font,size),count in fonts.items()):
        failures.append('No substantial standard 10-point body text detected')
    sheet=Image.new('RGB',(326*4,430*math.ceil(len(thumbs)/4)),'white')
    for i,thumb in enumerate(thumbs):
        sheet.paste(thumb,((i%4)*326,(i//4)*430))
    sheet.save(PREVIEW/'contact-sheet.png')
    (PAPER/'main.txt').write_text('\n\f\n'.join(text_parts))
    numeric=numerical_check()
    archive=ROOT/'paper/rewrite/upstream'
    if sha(archive/'main.pdf')!='ee983c7b5d0ba6e6c88e4f1f111c93f6f2a8078193ba624da8fa485766bb424c':
        failures.append('The preserved upstream PDF hash differs from the inspected input')
    if (ROOT/'paper.pdf').read_bytes()!=(PAPER/'main.pdf').read_bytes():
        failures.append('Root paper.pdf differs from the authoritative compiled PDF')
    record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'workflow_run':__import__('os').environ.get('GITHUB_RUN_ID'),'upstream_commit':PIN,
            'pages':len(doc),'format':'IEEEtran conference, two columns, 10-point body, US Letter',
            'pdf_sha256':sha(PAPER/'main.pdf'),'root_pdf_sha256':sha(ROOT/'paper.pdf'),
            'source_sha256':sha(PAPER/'main.tex'),'bibliography_sha256':sha(PAPER/'references.bib'),
            'upstream_pdf_sha256':sha(archive/'main.pdf'),
            'citation_keys':sorted(cited),'bibliography_entries':len(bibkeys),
            'numeric_verification':numeric,'unresolved_references':unresolved,'overfull_boxes':overfull,
            'page_layout':pages,'font_usage':[{'name':f,'size':s,'characters':n} for (f,s),n in fonts.most_common()],
            'source_evidence':json.loads((PAPER/'generated/source_ledger.json').read_text()),
            'failures':failures,'status':'passed' if not failures else 'failed',
            'research_status':'Reanalysis and writing only; no new detector training or full-dataset inference',
            'review_scope':'Automated numerical and PDF/layout checks; page previews are provided for separate visual review.'}
    (PAPER/'build.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print('PAPER VERIFICATION',json.dumps({k:v for k,v in record.items() if k not in ['source_evidence','font_usage']},indent=2))
    print('FONT USAGE',record['font_usage'])
    if failures:
        print('COMPILATION WARNINGS',log[-16000:])
        raise SystemExit(1)

if __name__=='__main__':
    main()
