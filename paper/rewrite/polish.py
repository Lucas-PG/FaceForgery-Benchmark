"""Apply final, exact-context editorial changes; no numerical edits.

The guarded build removes this one-time helper after committing the resulting
manuscript, bibliography, and figure-generation source.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/six-page'


def paragraph(text, prefix, replacement):
    lines = text.splitlines()
    indices = [i for i,line in enumerate(lines) if line.startswith(prefix)]
    if len(indices) != 1:
        raise ValueError('Unexpected editorial context: '+prefix)
    lines[indices[0]] = replacement
    return '\n'.join(lines)+'\n'


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected exact source context: '+old[:80])
    return text.replace(old,new)


def main():
    path = PAPER/'main.tex'
    text = path.read_text()
    text = paragraph(text,'The resulting conclusion is deliberately specific.','')
    text = paragraph(text,'This framing is consistent with WILDS,','')
    text = paragraph(text,'The external subset requires a qualification.',
        'The external preparation code samples selected available folders, pools authentic sources, and skips missing folders. Its evaluation manifest is absent from the aggregate release. We therefore use ``DF40 subset\'\' rather than claim complete official coverage. Transfer differences may reflect source composition and preprocessing as well as forgery methods; their interpretation is restricted to this local evaluation.')
    text = paragraph(text,'The present work adds analysis, not new detector training.',
        'The present work adds analysis, not new detector training. Numerical aggregation is reproducible, but historical predictions are not independently reproduced: checkpoint hashes, resolved configurations, external manifests, and per-example outputs are incomplete. The released code describes the intended pipeline rather than certifying every historical run. Original records remain unchanged, and the reconciliation identifies exclusions explicitly.')
    text = paragraph(text,'A narrower clean-to-degraded gap should not be read in isolation.',
        'A clean-to-degraded gap can shrink through better degraded discrimination, worse clean discrimination, or both. Table~\\ref{tab:primary} reports the absolute scores to distinguish these outcomes rather than equating a smaller gap with a better detector.')
    text = paragraph(text,'The pattern is compatible with adding weaker or redundant members',
        'The pattern is compatible with weaker or redundant members diluting a strong pair, but aggregate AUC does not identify that mechanism. A weak standalone detector may still repair important errors. Testing this possibility requires aligned per-image predictions and an analysis of repaired versus newly introduced errors at fixed thresholds; neither is inferred from ensemble size.')
    text = paragraph(text,'The representation matrix likewise favors an augmented RGB baseline',
        'The representation matrix favors establishing a strong augmented RGB baseline before adding complexity. This is conditional on the evaluated pipelines, not a rejection of tailored frequency-aware methods~\\cite{kashiani2025}. A controlled follow-up should test spectral information against identical augmentation, computation, and additional spatial capacity, without choosing variants by their best target-test row.')
    text = paragraph(text,'The analysis release includes immutable source identities,',
        'The analysis release includes source identities, all primary seed records, aggregation rules, generated tables, and the vector figure~\\cite{release2026}. Its fixed seed population is separate from secondary aggregates and qualitative illustrations. Original reports and the preceding manuscript remain available; incompatible cohorts and unverified label corrections are not silently combined into new empirical claims.')
    text = replace(text,' The implementation does not contain the H.264 or motion-blur operations sometimes associated with generic descriptions of robust preprocessing.','')
    text = replace(text,' Separate ensemble summaries favor a CLIP--DINOv3 pair over indiscriminate six-model fusion, but are not pooled with the primary seed cohort.','')
    text = replace(text,' These steps describe the evidence needed to explain the detection results; they do not claim that the explanation experiments have already established a mechanism.','')
    text = paragraph(text,'Generative AI assisted literature organization,',
        'ChatGPT (OpenAI) assisted literature organization, drafting and restructuring throughout the manuscript, and development of the scripts for Tables~I--V and Fig.~1. Primary numerical values are recomputed from released experiment records and independently checked by a separate aggregation implementation; no detector measurements or experimental observations were synthesized.')
    text = replace(text,'\\usepackage{microtype}','\\usepackage{microtype}\n\\usepackage{balance}')
    text = replace(text,'\\bibliographystyle{IEEEtran}','\\balance\n\\bibliographystyle{IEEEtran}')
    path.write_text(text)
    path = PAPER/'references.bib'
    text = path.read_text()
    abbreviations = {
        'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition':'Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)',
        'Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition':'Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)',
        'Proceedings of the IEEE/CVF International Conference on Computer Vision':'Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)',
        'Proceedings of the IEEE International Conference on Computer Vision':'Proc. IEEE Int. Conf. Comput. Vis. (ICCV)',
        'Proceedings of the 33rd ACM International Conference on Multimedia':'Proc. ACM Int. Conf. Multimedia (ACM MM)',
        'International Conference on Machine Learning':'Int. Conf. Mach. Learn. (ICML)',
        'International Conference on Learning Representations':'Int. Conf. Learn. Represent. (ICLR)',
        'Proceedings of Machine Learning Research':'Proc. Mach. Learn. Res.',
        'Advances in Neural Information Processing Systems':'Adv. Neural Inf. Process. Syst.'
    }
    for old,new in abbreviations.items():
        if old not in text:
            raise ValueError('Missing venue to abbreviate: '+old)
        text=text.replace(old,new)
    path.write_text(text)
    path=ROOT/'paper/rewrite/build_evidence.py'
    text=path.read_text()
    text=replace(text,'import matplotlib.pyplot as plt',"import matplotlib.pyplot as plt\nmatplotlib.rcParams['pdf.fonttype'] = 42")
    text=replace(text,"ax.legend(fontsize=7.7,loc='lower right',frameon=False)",
                 "ax.legend(fontsize=7.7,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False)")
    path.write_text(text)
    print('Final editorial and typography pass applied; numerical data and experiment records unchanged.')

if __name__=='__main__':
    main()
