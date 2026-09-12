"""Stage the six-page IEEE-format edition; leave the long reference untouched.

This is an editorial format conversion, not an experiment or evidence generator.
The generated source is committed separately after checking its compiled layout.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
reference = root / 'paper/explicability/main.tex'
output = root / 'paper/six-page'
output.mkdir(parents=True, exist_ok=True)
source = reference.read_text(encoding='utf-8')
body = source.split(r'\begin{abstract}', 1)[1]
body = r'\begin{abstract}' + body
# Operational appendices remain in the unchanged ten-page reference and guide.
body = body.split(r'\appendix', 1)[0]
body = body.replace(r'\citep', r'\cite')
body = body.replace(r'\citet{cunha2026}', r'Cunha et al.~\cite{cunha2026}')
body = body.replace(r'\bibliography{references}', r'\bibliography{../explicability/references}')
body = body.replace(r'\bibliographystyle{iclr2027_conference}', r'\bibliographystyle{IEEEtran}')
# IEEE column width requires a compact label in the inherited benchmark table.
body = body.replace('CLIP-style encoder, from scratch &', 'CLIP-style, scratch &')
body = body.replace('DINOv3, frozen configuration &', 'DINOv3, frozen &')
body = body.replace('Earlier configuration & Clean & Test-Hard & Absolute drop',
                    'Earlier configuration & Clean & Test-Hard & Drop')
body = body.replace('ViT, from scratch &', 'ViT, scratch &')
# Change author-review disclosure to an unnumbered conference-style section.
body = body.replace(r'\section*{AI assistance and author review}',
                    r'\section*{Disclosure and author review}')
preamble = r'''% Primary six-page edition. The ten-page reference remains in ../explicability/.
\documentclass[conference,10pt,letterpaper]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,url,array}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Explaining Spatial and Spectral Face Forgery Detectors},pdfauthor={Lucas Cunha; Lucas Sotomaior; Lucas Gasperin; Beatriz Caldas; Eduardo Pianovski; Rayson Laroca}}
\title{Explaining Spatial and Spectral Face Forgery Detectors:\\Controlled Comparisons and Shared Failure Analysis}
\author{\IEEEauthorblockN{Lucas Cunha\IEEEauthorrefmark{1}, Lucas Sotomaior\IEEEauthorrefmark{1}, Lucas Gasperin\IEEEauthorrefmark{1},\\Beatriz Caldas\IEEEauthorrefmark{1}, Eduardo Pianovski\IEEEauthorrefmark{1}, and Rayson Laroca\IEEEauthorrefmark{1}}
\IEEEauthorblockA{\IEEEauthorrefmark{1}Pontifical Catholic University of Paran\'{a}, Curitiba, Brazil\\
\IEEEauthorrefmark{1}\{c.oliveira25,lucas.sotomaior,lucas.gasperin,beatriz.caldas,eduardo.pianovski\}@pucpr.edu.br\\
\IEEEauthorrefmark{1}rayson@ppgia.pucpr.br}}
\newcommand{\AUC}{\operatorname{AUC}}
\newcommand{\BA}{\operatorname{BA}}
\newcommand{\ind}{\mathbb{1}}
\begin{document}
\maketitle
'''
(output / 'main.tex').write_text(preamble + body + '\n\\end{document}\n', encoding='utf-8')
print('Staged', output / 'main.tex')
print('Reference left unchanged:', reference)
