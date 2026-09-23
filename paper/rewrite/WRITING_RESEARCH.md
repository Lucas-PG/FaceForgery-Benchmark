# Writing research and editorial design

Research conducted 23 September 2026. These papers were read as primary sources, including introductions, experimental framing, limitations and selected figures. The manuscript is newly written around this project's actual released records; no wording or results are copied from the exemplars. Institutional prestige is not evidence that a scientific claim is valid.

## Stanford-led WILDS: A Benchmark of in-the-Wild Distribution Shifts

Koh et al., ICML 2021, PMLR 139, 5637–5664. Primary PDF: https://proceedings.mlr.press/v139/koh21a/koh21a.pdf

Read the abstract/introduction, opening benchmark overview and distribution-shift criteria. The paper establishes what counts as a shift before discussing scores, and separates benchmark design from algorithmic success. Applied here: define clean source test, its degraded counterpart and dataset transfer separately. Do not relabel MFFI's source-mixed clean test as i.i.d.; do not call a local DF40 subset the full official benchmark. The main question is whether an improvement transfers between evaluation conditions, not whether a model is vaguely robust.

## MIT: Adversarial Examples Are Not Bugs, They Are Features

Ilyas et al., NeurIPS 2019. Primary PDF: https://proceedings.neurips.cc/paper_files/paper/2019/file/e2c420d928d4bf8ce0ff2ec19b371514-Paper.pdf

Read the first-page argument and formal feature definitions. A concise, testable proposition organizes the paper, while the experiments and qualifications specify its reach. Applied here: make the mismatch between Test-D improvement and external transfer the central finding. Do not import the exemplar's causal conclusions into augmentation experiments; this project has neither its controlled feature construction nor a certified adversarial threat model.

## Harvard-associated interpretability research

Doshi-Velez and Kim, Towards a Rigorous Science of Interpretable Machine Learning, arXiv:1702.08608, 2017, a position paper rather than a claimed conference acceptance. Primary PDF: https://arxiv.org/pdf/1702.08608 . Harvard's course explicitly discusses this paper: https://embeddedethics.seas.harvard.edu/cs-252r-fall-2020/ . Author's institutional site: https://finale.seas.harvard.edu/ .

Read the definition/motivation and evaluation taxonomy. Explanations, safety, fairness and causality are distinct claims with different evidence requirements. Applied here: treat the existing two-image gallery as an illustration, not quantitative explanation evaluation or proof of semantic reasoning. State what controlled same-image analysis would test without presenting unexecuted protocols as findings.

## OpenAI: Learning Transferable Visual Models From Natural Language Supervision

Radford et al., ICML 2021, PMLR 139, 8748–8763. Primary PDF: https://proceedings.mlr.press/v139/radford21a/radford21a.pdf

Read the opening problem-to-method transition and robustness/limitations discussion; inspected the robustness plot and comparison framing. Strong claims are attached to named tasks and matched comparators, and development-set reuse is disclosed. Applied here: put scope and seed count in every result table, distinguish seed dispersion from confidence intervals, and avoid presenting CLIP's original zero-shot results as a matched baseline for our fine-tuned CLIP classifier.

## Berkeley/Adobe/Michigan: CNN-generated Images Are Surprisingly Easy to Spot... for Now

Wang et al., CVPR 2020. Primary PDF: https://arxiv.org/pdf/1912.11035 . Official entry: https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html

Read the introduction and visual overview of the training/generalization experiment. A simple baseline and carefully described training variation make the empirical question concrete; the title bounds the result. Applied here: describe the actual JPEG/blur/noise augmentation rather than an invented H.264 pipeline, and distinguish the tested FFT encodings from all possible frequency-aware detectors.

## Wisconsin–Madison: Towards Universal Fake Image Detectors that Generalize Across Generative Models

Ojha, Li and Lee, CVPR 2023, 24480–24489. Primary PDF: https://arxiv.org/pdf/2302.10174 .

Read the opening diagnosis and baseline formulation. The narrative moves from a precise failure to a small alternative and a testable comparison. Applied here: establish the released RGB baseline before spectral inputs or larger ensembles; place negative results and exceptions beside the main pattern. The exemplar's frozen-feature/linear-probe setup is not equated with this project's fine-tuning.

## Microsoft Research: Deep Residual Learning for Image Recognition

He et al., CVPR 2016. Primary institutional source: https://www.microsoft.com/en-us/research/publication/deep-residual-learning-for-image-recognition/ .

Read the motivating optimization problem and contrast between the proposed mechanism and empirical test. Applied here: separate detector families from explanations of why their results differ. Report backbone-conditioned associations; do not label a saliency image or AUC difference as proof of a learned mechanism.

## Contemporary field comparator: FreqDebias

Kashiani, Talemi and Afghah, CVPR 2025, 8775–8785. Proceedings: https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html . Full author PDF: https://arxiv.org/pdf/2509.22412 .

Read Fo-Mixup and the component-ablation section; inspected the ablation page. The experiment distinguishes augmentation, consistency and selection mechanisms. Applied here: acknowledge non-factorial historical augmentation policies and avoid treating naive magnitude/phase inputs as a reproduction or refutation of FreqDebias. No unmatched published score is inserted into our empirical leaderboard.

## Additional scientific source checks

MFFI's original paper (https://arxiv.org/pdf/2509.05592) was read for split design and degradation operations, including Table 2 and Section 3.5. Its clean split already varies authentic sources; Test-D includes conventional disturbances and patch-based adversarial perturbations. Neither is reduced to a compression-only benchmark. DeepfakeBench motivates harmonized processing and evaluation (https://proceedings.neurips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html). DF40's official project (https://github.com/YZY-stack/DF40) defines the full benchmark, distinct from the local export analyzed here. Celeb-DF's proceedings entry (https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.html) establishes the video-forensics context, not the polarity or sample coverage of our historical local export.

The preceding project paper was verified at https://arxiv.org/abs/2609.01511. This rewrite cites the earlier study rather than presenting its results as new. DINOv3 was checked at https://arxiv.org/abs/2508.10104; the repository's actual DINO configuration is ConvNeXt-Base, not a DINO vision transformer or Tiny backbone.

## Editorial contract

The abstract gives the question, exact primary experimental population, quantitative observations and a bounded implication. The introduction establishes one coherent contribution rather than a catalog of scripts. Related work is organized by the comparison it changes. Methods separate benchmark definition, released implementation, recovered experimental metadata and retrospective reanalysis. Results use declarative subsection headings, generated tables, named denominators and negative findings. Limitations identify which conclusions survive incomplete provenance. The conclusion answers the opening question without a new score or acceptance promise.

No institutional names are inserted into the article to confer authority. No claims of novelty, causal explanation, statistical significance, unseen-person independence or deployment readiness are added without evidence. Author names, affiliation and order are preserved. Primary table values are regenerated from pinned per-seed files, never copied from the previous prose or screenshots.
