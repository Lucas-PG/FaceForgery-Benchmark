# Sumário Geral e Tabela Consolidada de Resultados

> Este repositório documenta a avaliação empírica de **6 famílias arquiteturais de redes neurais**, **modelos Mixture of Experts (MoE)** e **comitês (Ensembles)** sob **7 regimes de representação no domínio de frequência (2D-FFT)** contra imagens autênticas e manipuladas por Deepfakes.

## 1. Tabela de Classificação Global de Robustez (Test vs Test_d)

Classificação de modelos individuais e ensembles ordenados pelo desempenho em dados com perturbações adversas e compressão (`test_d`).

| Posição | Família do Modelo | Modo Fourier / Fusão | Regime | Sementes | Test AUC | Test ACC | Test_d AUC | Test_d ACC | Queda Δ AUC | Score Geral | Conceito |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ENSEMBLE (4M)** | `top4_hybrid [mean]` | `ensemble` | 5 | 0.9578 | 0.8945 | **0.7800** | 0.7038 | `-0.1778` | 7.73 | **A** |
| 2 | **ENSEMBLE (4M)** | `top4_hybrid [weighted]` | `ensemble` | 5 | 0.9582 | 0.8945 | **0.7799** | 0.7046 | `-0.1783` | 7.73 | **A** |
| 3 | **ENSEMBLE (5M)** | `top5_full [weighted]` | `ensemble` | 5 | 0.9552 | 0.8823 | **0.7796** | 0.7045 | `-0.1756` | 7.72 | **A** |
| 4 | **ENSEMBLE (5M)** | `top5_full [mean]` | `ensemble` | 5 | 0.9544 | 0.8810 | **0.7794** | 0.7039 | `-0.1750` | 7.71 | **A** |
| 5 | **ENSEMBLE (3M)** | `top3_diverse [weighted]` | `ensemble` | 5 | 0.9578 | 0.8907 | **0.7759** | 0.6991 | `-0.1819` | 7.71 | **A** |
| 6 | **ENSEMBLE (3M)** | `top3_diverse [mean]` | `ensemble` | 5 | 0.9570 | 0.8903 | **0.7757** | 0.6980 | `-0.1813` | 7.71 | **A** |
| 7 | **ENSEMBLE (3M)** | `top3_diverse [geometric]` | `ensemble` | 5 | 0.9601 | 0.8411 | **0.7740** | 0.6713 | `-0.1861` | 7.71 | **A** |
| 8 | **ENSEMBLE (4M)** | `top4_hybrid [geometric]` | `ensemble` | 5 | 0.9593 | 0.8379 | **0.7735** | 0.6708 | `-0.1859` | 7.70 | **A** |
| 9 | **ENSEMBLE (5M)** | `top5_full [stacking]` | `ensemble` | 5 | 0.9614 | 0.8946 | **0.7734** | 0.7066 | `-0.1880` | 7.71 | **A** |
| 10 | **ENSEMBLE (4M)** | `top4_hybrid [stacking]` | `ensemble` | 5 | 0.9614 | 0.8949 | **0.7727** | 0.7062 | `-0.1886` | 7.71 | **A** |
| 11 | **ENSEMBLE (5M)** | `top5_full [geometric]` | `ensemble` | 5 | 0.9551 | 0.8089 | **0.7717** | 0.6611 | `-0.1834` | 7.68 | **A** |
| 12 | **ENSEMBLE (3M)** | `top3_absolute [mean]` | `ensemble` | 5 | 0.9562 | 0.8864 | **0.7708** | 0.7081 | `-0.1855` | 7.68 | **A** |
| 13 | **ENSEMBLE (3M)** | `top3_absolute [weighted]` | `ensemble` | 5 | 0.9565 | 0.8866 | **0.7707** | 0.7087 | `-0.1858` | 7.68 | **A** |
| 14 | **ENSEMBLE (3M)** | `top3_diverse [stacking]` | `ensemble` | 5 | 0.9605 | 0.8928 | **0.7705** | 0.7011 | `-0.1901` | 7.69 | **A** |
| 15 | **ENSEMBLE (3M)** | `top3_absolute [geometric]` | `ensemble` | 5 | 0.9586 | 0.8521 | **0.7667** | 0.6756 | `-0.1919` | 7.67 | **A** |
| 16 | **ENSEMBLE (3M)** | `top3_absolute [stacking]` | `ensemble` | 5 | 0.9597 | 0.8882 | **0.7661** | 0.6868 | `-0.1936` | 7.67 | **A** |
| 17 | **ENSEMBLE (3M)** | `top3_absolute [max]` | `ensemble` | 5 | 0.9493 | 0.8804 | **0.7632** | 0.6794 | `-0.1861` | 7.61 | **A** |
| 18 | **ENSEMBLE (3M)** | `val_optimal_search [stacking]` | `ensemble` | 5 | 0.9590 | 0.8860 | **0.7570** | 0.6835 | `-0.2020` | 7.62 | **A** |
| 19 | **ENSEMBLE (3M)** | `top3_diverse [max]` | `ensemble` | 5 | 0.9442 | 0.8437 | **0.7568** | 0.6776 | `-0.1874` | 7.56 | **A** |
| 20 | **ENSEMBLE (3M)** | `top3_diverse [optimal_weights]` | `ensemble` | 5 | 0.9581 | 0.8597 | **0.7521** | 0.6623 | `-0.2060` | 7.59 | **A** |
| 21 | **ENSEMBLE (3M)** | `top3_absolute [optimal_weights]` | `ensemble` | 5 | 0.9581 | 0.8597 | **0.7521** | 0.6623 | `-0.2060` | 7.59 | **A** |
| 22 | **ENSEMBLE (5M)** | `top5_full [optimal_weights]` | `ensemble` | 5 | 0.9581 | 0.8596 | **0.7520** | 0.6623 | `-0.2061` | 7.59 | **A** |
| 23 | **ENSEMBLE (4M)** | `top4_hybrid [optimal_weights]` | `ensemble` | 5 | 0.9581 | 0.8596 | **0.7520** | 0.6622 | `-0.2061` | 7.59 | **A** |
| 24 | **ENSEMBLE (3M)** | `val_optimal_search [optimal_weights]` | `ensemble` | 5 | 0.9584 | 0.8593 | **0.7511** | 0.6620 | `-0.2073` | 7.59 | **A** |
| 25 | **ENSEMBLE (5M)** | `top5_full [max]` | `ensemble` | 5 | 0.9448 | 0.8167 | **0.7502** | 0.6690 | `-0.1946` | 7.53 | **A** |
| 26 | **ENSEMBLE (4M)** | `top4_hybrid [max]` | `ensemble` | 5 | 0.9456 | 0.8291 | **0.7498** | 0.6711 | `-0.1959` | 7.53 | **A** |
| 27 | **ENSEMBLE (3M)** | `val_optimal_search [geometric]` | `ensemble` | 5 | 0.9553 | 0.8168 | **0.7498** | 0.6411 | `-0.2055` | 7.57 | **A** |
| 28 | **clip** | `none` | `finetune` | 5 | 0.9195 | 0.8364 | **0.7489** | 0.6814 | `-0.1706` | 7.41 | **A** |
| 29 | **ENSEMBLE (3M)** | `val_optimal_search [mean]` | `ensemble` | 5 | 0.9551 | 0.8639 | **0.7483** | 0.6645 | `-0.2067` | 7.56 | **A** |
| 30 | **ENSEMBLE (3M)** | `val_optimal_search [weighted]` | `ensemble` | 5 | 0.9551 | 0.8639 | **0.7483** | 0.6645 | `-0.2068` | 7.56 | **A** |
| 31 | **ENSEMBLE (3M)** | `val_optimal_search [max]` | `ensemble` | 5 | 0.9487 | 0.8854 | **0.7408** | 0.6775 | `-0.2079` | 7.50 | **A** |
| 32 | **clip** | `concat` | `finetune` | 5 | 0.8941 | 0.8151 | **0.7260** | 0.6778 | `-0.1681` | 7.19 | **A** |
| 33 | **dino** | `none` | `finetune` | 5 | 0.9358 | 0.8763 | **0.7147** | 0.6680 | `-0.2211` | 7.30 | **A** |
| 34 | **vit** | `concat` | `finetune` | 5 | 0.8265 | 0.7569 | **0.7123** | 0.6601 | `-0.1142` | 6.86 | **B+** |
| 35 | **ENSEMBLE (3M)** | `top3_absolute [majority]` | `ensemble` | 5 | 0.8890 | 0.8805 | **0.6976** | 0.7055 | `-0.1914` | 7.04 | **A** |
| 36 | **ENSEMBLE (5M)** | `top5_full [majority]` | `ensemble` | 5 | 0.8821 | 0.8726 | **0.6958** | 0.7013 | `-0.1863` | 7.01 | **A** |
| 37 | **vit** | `none` | `finetune` | 5 | 0.8143 | 0.7307 | **0.6925** | 0.6330 | `-0.1218` | 6.70 | **B+** |
| 38 | **ENSEMBLE (3M)** | `top3_diverse [majority]` | `ensemble` | 5 | 0.8898 | 0.8812 | **0.6866** | 0.6934 | `-0.2032` | 6.99 | **B+** |
| 39 | **resnet** | `concat_frequency` | `finetune` | 5 | 0.8337 | 0.7469 | **0.6866** | 0.6402 | `-0.1472` | 6.75 | **B+** |
| 40 | **dino** | `concat` | `finetune` | 5 | 0.9166 | 0.8512 | **0.6834** | 0.6384 | `-0.2332` | 7.07 | **A** |
| 41 | **dino** | `concat_frequency` | `finetune` | 5 | 0.8603 | 0.7825 | **0.6820** | 0.6629 | `-0.1783` | 6.83 | **B+** |
| 42 | **MoE (7 experts) - moe_standard** | `none` | `scratch` | 1 | 0.8773 | 0.7773 | **0.6796** | 0.6199 | `-0.1976` | 7.59 | **A** |
| 43 | **ENSEMBLE (4M)** | `top4_hybrid [majority]` | `ensemble` | 5 | 0.8957 | 0.8948 | **0.6770** | 0.6987 | `-0.2188` | 6.97 | **B+** |
| 44 | **vit** | `concat_frequency` | `finetune` | 5 | 0.7915 | 0.7166 | **0.6756** | 0.6443 | `-0.1159` | 6.53 | **B+** |
| 45 | **ENSEMBLE (3M)** | `val_optimal_search [majority]` | `ensemble` | 5 | 0.8720 | 0.8587 | **0.6736** | 0.6619 | `-0.1984` | 6.86 | **B+** |
| 46 | **mobilenet** | `none` | `finetune` | 5 | 0.8470 | 0.7424 | **0.6700** | 0.6176 | `-0.1770` | 6.73 | **B+** |
| 47 | **resnet** | `concat` | `finetune` | 5 | 0.8717 | 0.7657 | **0.6662** | 0.6047 | `-0.2055` | 6.78 | **B+** |
| 48 | **resnet** | `none` | `finetune` | 5 | 0.8845 | 0.7771 | **0.6647** | 0.6119 | `-0.2197` | 6.84 | **B+** |
| 49 | **MoE (7 experts) - moe_frequency** | `concat_frequency` | `scratch` | 1 | 0.9068 | 0.8253 | **0.6512** | 0.6240 | `-0.2556` | 7.53 | **A** |
| 50 | **clip** | `concat_frequency` | `finetune` | 5 | 0.7789 | 0.7057 | **0.6505** | 0.6265 | `-0.1284` | 6.35 | **B** |
| 51 | **xception** | `none` | `finetune` | 5 | 0.7729 | 0.7059 | **0.6297** | 0.6045 | `-0.1432` | 6.23 | **B** |
| 52 | **mobilenet** | `concat` | `finetune` | 5 | 0.8049 | 0.7184 | **0.6296** | 0.5940 | `-0.1752` | 6.36 | **B** |
| 53 | **xception** | `concat` | `finetune` | 5 | 0.7329 | 0.6720 | **0.6119** | 0.5904 | `-0.1210` | 5.98 | **C** |
| 54 | **mobilenet** | `concat_frequency` | `finetune` | 5 | 0.7084 | 0.6394 | **0.6032** | 0.5674 | `-0.1052` | 5.84 | **C** |
| 55 | **xception** | `concat_frequency` | `finetune` | 5 | 0.6592 | 0.6101 | **0.5746** | 0.5754 | `-0.0846` | 5.50 | **C** |
| 56 | **vit** | `complex` | `finetune` | 5 | 0.5701 | 0.5735 | **0.5541** | 0.5735 | `-0.0160` | 5.04 | **D** |
| 57 | **mobilenet** | `frequency_3` | `finetune` | 5 | 0.7127 | 0.6510 | **0.5452** | 0.5228 | `-0.1676` | 5.56 | **C** |
| 58 | **resnet** | `complex` | `finetune` | 5 | 0.6992 | 0.6415 | **0.5436** | 0.5428 | `-0.1556` | 5.47 | **D** |
| 59 | **mobilenet** | `magnitude` | `finetune` | 5 | 0.7177 | 0.6592 | **0.5420** | 0.5265 | `-0.1757` | 5.57 | **C** |
| 60 | **resnet** | `magnitude` | `finetune` | 5 | 0.7503 | 0.6855 | **0.5409** | 0.5469 | `-0.2094` | 5.68 | **C** |
| 61 | **dino** | `frequency_3` | `finetune` | 5 | 0.7631 | 0.6821 | **0.5371** | 0.5418 | `-0.2260` | 5.72 | **C** |
| 62 | **resnet** | `frequency_3` | `finetune` | 5 | 0.7500 | 0.6855 | **0.5365** | 0.5414 | `-0.2135` | 5.67 | **C** |
| 63 | **dino** | `magnitude` | `finetune` | 5 | 0.7614 | 0.6778 | **0.5365** | 0.5460 | `-0.2249` | 5.72 | **C** |
| 64 | **vit** | `magnitude` | `finetune` | 5 | 0.7240 | 0.6355 | **0.5347** | 0.5497 | `-0.1893` | 5.55 | **C** |
| 65 | **mobilenet** | `complex` | `finetune` | 5 | 0.6872 | 0.6165 | **0.5337** | 0.5215 | `-0.1534` | 5.41 | **D** |
| 66 | **dino** | `phase` | `finetune` | 5 | 0.7128 | 0.6519 | **0.5332** | 0.5616 | `-0.1795` | 5.50 | **C** |
| 67 | **xception** | `frequency_3` | `finetune` | 5 | 0.6994 | 0.6485 | **0.5330** | 0.5382 | `-0.1664` | 5.44 | **D** |
| 68 | **clip** | `frequency_3` | `finetune` | 5 | 0.7316 | 0.6713 | **0.5328** | 0.5309 | `-0.1988` | 5.58 | **C** |
| 69 | **xception** | `magnitude` | `finetune` | 5 | 0.7035 | 0.6514 | **0.5328** | 0.5496 | `-0.1707` | 5.46 | **D** |
| 70 | **clip** | `magnitude` | `finetune` | 5 | 0.7336 | 0.6732 | **0.5300** | 0.5440 | `-0.2036` | 5.58 | **C** |
| 71 | **vit** | `frequency_3` | `finetune` | 5 | 0.7240 | 0.6310 | **0.5281** | 0.5561 | `-0.1959` | 5.52 | **C** |
| 72 | **mobilenet** | `phase` | `finetune` | 5 | 0.6287 | 0.5545 | **0.5234** | 0.5001 | `-0.1052` | 5.12 | **D** |
| 73 | **resnet** | `phase` | `finetune` | 5 | 0.6467 | 0.6070 | **0.5211** | 0.5403 | `-0.1256` | 5.18 | **D** |
| 74 | **clip** | `complex` | `finetune` | 5 | 0.5271 | 0.5735 | **0.5197** | 0.5735 | `-0.0074` | 4.64 | **F** |
| 75 | **xception** | `complex` | `finetune` | 5 | 0.5622 | 0.5735 | **0.5174** | 0.5734 | `-0.0448` | 4.83 | **F** |
| 76 | **xception** | `phase` | `finetune` | 5 | 0.5889 | 0.5808 | **0.5111** | 0.5691 | `-0.0778` | 4.90 | **F** |
| 77 | **clip** | `phase` | `finetune` | 5 | 0.6265 | 0.5879 | **0.4994** | 0.5358 | `-0.1272` | 4.99 | **F** |
| 78 | **dino** | `complex` | `finetune` | 5 | 0.4995 | 0.5735 | **0.4977** | 0.5734 | `-0.0018` | 4.48 | **F** |
| 79 | **vit** | `phase` | `finetune` | 5 | 0.6020 | 0.5740 | **0.4955** | 0.5734 | `-0.1064` | 4.88 | **F** |

## 2. Tabela de Campeões por Família Arquitetural

| Família Arquitetural | Melhor Modo Fourier | Test AUC | Test ACC | Test_d AUC | Test_d ACC | Δ AUC (Robustez) | DF40 AUC | Celeb-DF Frame AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP** | `none` | 0.9195 ± 0.0041 | 0.8364 ± 0.0072 | **0.7489 ± 0.0103** | 0.6814 ± 0.0041 | `-0.1706 ± 0.0085` | 0.7639 ± 0.0460 | 0.3324 ± 0.0310 |
| **DINO** | `none` | 0.9358 ± 0.0062 | 0.8763 ± 0.0085 | **0.7147 ± 0.0115** | 0.6680 ± 0.0098 | `-0.2211 ± 0.0099` | 0.7208 ± 0.0289 | 0.3355 ± 0.0600 |
| **VIT** | `concat` | 0.8265 ± 0.0035 | 0.7569 ± 0.0049 | **0.7123 ± 0.0056** | 0.6601 ± 0.0053 | `-0.1142 ± 0.0036` | 0.6324 ± 0.0237 | 0.4519 ± 0.0146 |
| **RESNET** | `concat_frequency` | 0.8337 ± 0.0072 | 0.7469 ± 0.0085 | **0.6866 ± 0.0103** | 0.6402 ± 0.0124 | `-0.1472 ± 0.0048` | 0.7071 ± 0.0316 | 0.4344 ± 0.0231 |
| **MOBILENET** | `none` | 0.8470 ± 0.0024 | 0.7424 ± 0.0031 | **0.6700 ± 0.0052** | 0.6176 ± 0.0022 | `-0.1770 ± 0.0058` | 0.7037 ± 0.0140 | 0.3804 ± 0.0115 |
| **XCEPTION** | `none` | 0.7729 ± 0.0042 | 0.7059 ± 0.0027 | **0.6297 ± 0.0013** | 0.6045 ± 0.0016 | `-0.1432 ± 0.0046` | 0.7298 ± 0.0244 | 0.3979 ± 0.0144 |
| **MoE (Frequencial, 7 exp)** | `concat_frequency` | `0.9068` | `0.8253` | **`0.6512`** | `0.6240` | `-0.2556` | Em andamento | Em andamento |
| **MoE (Espacial, 7 exp)** | `none` | `0.8773` | `0.7773` | **`0.6796`** | `0.6199` | `-0.1976` | Em andamento | Em andamento |

## 3. Tabela de Impacto da Representação Frequencial (Espacial vs Híbrido Fourier)

Comparação direta entre o modelo de linha de base puramente espacial (`none`), o modelo híbrido concatenado (`concat`) e a fusão frequencial pura (`concat_frequency`).

| Família | Test AUC (`none`) | Test AUC (`concat`) | Test AUC (`concat_freq`) | Test_d AUC (`none`) | Test_d AUC (`concat`) | Test_d AUC (`concat_freq`) | Domínio Mais Robusto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP** | 0.9195 | 0.8941 | 0.7789 | 0.7489 | 0.7260 | 0.6505 | none (Espacial) |
| **DINO** | 0.9358 | 0.9166 | 0.8603 | 0.7147 | 0.6834 | 0.6820 | none (Espacial) |
| **VIT** | 0.8143 | 0.8265 | 0.7915 | 0.6925 | 0.7123 | 0.6756 | **concat (Híbrido)** |
| **RESNET** | 0.8845 | 0.8717 | 0.8337 | 0.6647 | 0.6662 | 0.6866 | **concat_freq (Fourier)** |
| **MOBILENET** | 0.8470 | 0.8049 | 0.7084 | 0.6700 | 0.6296 | 0.6032 | none (Espacial) |
| **XCEPTION** | 0.7729 | 0.7329 | 0.6592 | 0.6297 | 0.6119 | 0.5746 | none (Espacial) |

## 4. Índice e Navegação dos Relatórios de Modelos

| Relatório (.md) | Família / Categoria | Número de Especialistas / Variantes | Métricas Contidas |
| :--- | :--- | :---: | :--- |
| [`moe_frequency.md`](./moe_frequency.md) | Mixture of Experts Frequencial | 7 Especialistas | Val, Test, Test_d, Evolução 4 exp vs 7 exp, Alocação de Especialistas |
| [`moe_standard.md`](./moe_standard.md) | Mixture of Experts Padrão | 7 Especialistas | Val, Test, Test_d, Comparativo com MoE Frequencial, Evolução 4 exp vs 7 exp |
| [`moe_4experts.md`](./moe_4experts.md) | MoE Histórico (Linha de Base) | 4 Especialistas | Val, Test, Comparativo Frequency vs Standard |
| [`clip.md`](./clip.md) | CLIP ViT-B/16 | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |
| [`dino.md`](./dino.md) | DINO ViT-S/16 | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |
| [`vit.md`](./vit.md) | Vision Transformer (ViT-B/16) | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |
| [`resnet.md`](./resnet.md) | ResNet-50 | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |
| [`mobilenet.md`](./mobilenet.md) | MobileNetV3-Small | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |
| [`xception.md`](./xception.md) | XceptionNet | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |
| [`ensembles.md`](./ensembles.md) | Comitês de Modelos (Ensembles) | Top-4, Top-5, Top-3 e Val Optimal | Fusões Ponderadas, Médias, Stacking, Celeb-DF e DF40 |
