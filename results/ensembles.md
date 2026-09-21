# Resultados dos Modelos em Ensemble (Comitês)

> **Visão Geral**: Avaliação sistemática de arquiteturas combinadas por fusão probabilística, votação e empilhamento (stacking), visando maximizar a robustez e mitigar o overfitting aos artefatos de treino.

> [!NOTE]
> - Para a suíte oficial de **Ensembles Baseline** ($K=2..6$ ordenados por Val AUC), consulte [`results/mostrar_rayson/tabela2-resultados-ensemble.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela2-resultados-ensemble.md).
> - Para os novos **Ensembles de Modelos Robustos** com `RandomizedRobustAugment`, consulte [`results/mostrar_rayson/tabela7-ensemble-robusto.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela7-ensemble-robusto.md).

## 1. Tabela Principal de Desempenho dos Ensembles no Teste Padrão e Teste Difícil

| Ranking | ID do Ensemble | Composição | Estratégia de Fusão | Val AUC | Test AUC | Test ACC | Test F1 | Test_d AUC | Test_d ACC | Test_d F1 | Δ AUC | Score | Conceito |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `weighted` | 0.9972 | 0.9582 | 0.8945 | 0.9023 | **0.7799** | 0.7046 | 0.7470 | `-0.1783` | 7.73 | **A** |
| 2 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `mean` | 0.9970 | 0.9578 | 0.8945 | 0.9024 | **0.7800** | 0.7038 | 0.7472 | `-0.1778` | 7.73 | **A** |
| 3 | `top5_full` | Top-5 Completo | `weighted` | 0.9966 | 0.9552 | 0.8823 | 0.8897 | **0.7796** | 0.7045 | 0.7407 | `-0.1756` | 7.72 | **A** |
| 4 | `top5_full` | Top-5 Completo | `stacking` | 0.9987 | 0.9614 | 0.8946 | 0.9017 | **0.7734** | 0.7066 | 0.7337 | `-0.1880` | 7.71 | **A** |
| 5 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `stacking` | 0.9987 | 0.9614 | 0.8949 | 0.9020 | **0.7727** | 0.7062 | 0.7336 | `-0.1886` | 7.71 | **A** |
| 6 | `top5_full` | Top-5 Completo | `mean` | 0.9964 | 0.9544 | 0.8810 | 0.8884 | **0.7794** | 0.7039 | 0.7406 | `-0.1750` | 7.71 | **A** |
| 7 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `weighted` | 0.9977 | 0.9578 | 0.8907 | 0.8980 | **0.7759** | 0.6991 | 0.7365 | `-0.1819` | 7.71 | **A** |
| 8 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `mean` | 0.9975 | 0.9570 | 0.8903 | 0.8978 | **0.7757** | 0.6980 | 0.7367 | `-0.1813` | 7.71 | **A** |
| 9 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `geometric` | 0.9982 | 0.9601 | 0.8411 | 0.8417 | **0.7740** | 0.6713 | 0.6712 | `-0.1861` | 7.71 | **A** |
| 10 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `geometric` | 0.9979 | 0.9593 | 0.8379 | 0.8379 | **0.7735** | 0.6708 | 0.6726 | `-0.1859` | 7.70 | **A** |
| 11 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `stacking` | 0.9986 | 0.9605 | 0.8928 | 0.8999 | **0.7705** | 0.7011 | 0.7287 | `-0.1901` | 7.69 | **A** |
| 12 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `mean` | 0.9971 | 0.9562 | 0.8864 | 0.8940 | **0.7708** | 0.7081 | 0.7455 | `-0.1855` | 7.68 | **A** |
| 13 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `weighted` | 0.9972 | 0.9565 | 0.8866 | 0.8942 | **0.7707** | 0.7087 | 0.7457 | `-0.1858` | 7.68 | **A** |
| 14 | `top5_full` | Top-5 Completo | `geometric` | 0.9973 | 0.9551 | 0.8089 | 0.8029 | **0.7717** | 0.6611 | 0.6543 | `-0.1834` | 7.68 | **A** |
| 15 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `geometric` | 0.9981 | 0.9586 | 0.8521 | 0.8545 | **0.7667** | 0.6756 | 0.6799 | `-0.1919` | 7.67 | **A** |
| 16 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `stacking` | 0.9986 | 0.9597 | 0.8882 | 0.8950 | **0.7661** | 0.6868 | 0.7068 | `-0.1936` | 7.67 | **A** |
| 17 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `stacking` | 0.9987 | 0.9590 | 0.8860 | 0.8931 | **0.7570** | 0.6835 | 0.7035 | `-0.2020` | 7.62 | **A** |
| 18 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `max` | 0.9911 | 0.9493 | 0.8804 | 0.8984 | **0.7632** | 0.6794 | 0.7577 | `-0.1861` | 7.61 | **A** |
| 19 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `optimal_weights` | 0.9985 | 0.9581 | 0.8597 | 0.8638 | **0.7521** | 0.6623 | 0.6634 | `-0.2060` | 7.59 | **A** |
| 20 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `optimal_weights` | 0.9985 | 0.9581 | 0.8597 | 0.8638 | **0.7521** | 0.6623 | 0.6634 | `-0.2060` | 7.59 | **A** |
| 21 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `optimal_weights` | 0.9985 | 0.9581 | 0.8596 | 0.8638 | **0.7520** | 0.6622 | 0.6633 | `-0.2061` | 7.59 | **A** |
| 22 | `top5_full` | Top-5 Completo | `optimal_weights` | 0.9985 | 0.9581 | 0.8596 | 0.8638 | **0.7520** | 0.6623 | 0.6633 | `-0.2061` | 7.59 | **A** |
| 23 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `optimal_weights` | 0.9985 | 0.9584 | 0.8593 | 0.8634 | **0.7511** | 0.6620 | 0.6629 | `-0.2073` | 7.59 | **A** |
| 24 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `geometric` | 0.9983 | 0.9553 | 0.8168 | 0.8131 | **0.7498** | 0.6411 | 0.6244 | `-0.2055` | 7.57 | **A** |
| 25 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `weighted` | 0.9983 | 0.9551 | 0.8639 | 0.8691 | **0.7483** | 0.6645 | 0.6718 | `-0.2068` | 7.56 | **A** |
| 26 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `mean` | 0.9983 | 0.9551 | 0.8639 | 0.8692 | **0.7483** | 0.6645 | 0.6718 | `-0.2067` | 7.56 | **A** |
| 27 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `max` | 0.9900 | 0.9442 | 0.8437 | 0.8741 | **0.7568** | 0.6776 | 0.7635 | `-0.1874` | 7.56 | **A** |
| 28 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `max` | 0.9882 | 0.9456 | 0.8291 | 0.8651 | **0.7498** | 0.6711 | 0.7645 | `-0.1959` | 7.53 | **A** |
| 29 | `top5_full` | Top-5 Completo | `max` | 0.9881 | 0.9448 | 0.8167 | 0.8571 | **0.7502** | 0.6690 | 0.7645 | `-0.1946` | 7.53 | **A** |
| 30 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `max` | 0.9962 | 0.9487 | 0.8854 | 0.9011 | **0.7408** | 0.6775 | 0.7489 | `-0.2079` | 7.50 | **A** |
| 31 | `top3_absolute` | Top-3 Absoluto (Maiores Notas) | `majority` | 0.9633 | 0.8890 | 0.8805 | 0.8887 | **0.6976** | 0.7055 | 0.7454 | `-0.1914` | 7.04 | **A** |
| 32 | `top5_full` | Top-5 Completo | `majority` | 0.9638 | 0.8821 | 0.8726 | 0.8803 | **0.6958** | 0.7013 | 0.7378 | `-0.1863` | 7.01 | **A** |
| 33 | `top3_diverse` | Top-3 Diverso (ViT + ConvNeXt + CNN Fourier) | `majority` | 0.9695 | 0.8898 | 0.8812 | 0.8892 | **0.6866** | 0.6934 | 0.7329 | `-0.2032` | 6.99 | **B+** |
| 34 | `top4_hybrid` | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `majority` | 0.9478 | 0.8957 | 0.8948 | 0.9065 | **0.6770** | 0.6987 | 0.7584 | `-0.2188` | 6.97 | **B+** |
| 35 | `val_optimal_search` | Seleção Ótima no Val (3 modelos) | `majority` | 0.9782 | 0.8720 | 0.8587 | 0.8639 | **0.6736** | 0.6619 | 0.6683 | `-0.1984` | 6.86 | **B+** |

## 2. Generalização Cross-Dataset no Celeb-DF (In-the-Wild)

| Ensemble | Estratégia de Fusão | Frame AUC (%) | Frame ACC (%) | Video AUC (%) | Video ACC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DINO + XCEPTION + VIT** | `max` | 38.85% | 33.75% | **34.11%** | 33.98% |
| **CLIP + DINO** | `max` | 37.35% | 33.27% | **31.99%** | 34.17% |
| **CLIP + DINO + VIT** | `max` | 37.29% | 33.28% | **31.78%** | 33.59% |
| **CLIP + DINO + XCEPTION** | `max` | 37.40% | 33.81% | **31.68%** | 33.98% |
| **CLIP + DINO + RESNET** | `max` | 35.67% | 33.28% | **31.36%** | 33.78% |
| **DINO + XCEPTION + VIT** | `mean` | 35.94% | 32.97% | **31.26%** | 30.89% |
| **CLIP + DINO** | `mean` | 35.63% | 32.16% | **31.00%** | 32.05% |
| **DINO + XCEPTION + VIT** | `geom` | 35.60% | 36.87% | **30.99%** | 34.75% |
| **TODOS OS 6 ROBUSTOS** | `max` | 35.26% | 34.04% | **30.54%** | 34.36% |
| **CLIP + DINO + VIT** | `mean` | 35.11% | 32.79% | **30.50%** | 30.31% |
| **CLIP + DINO** | `geom` | 35.40% | 34.09% | **30.43%** | 31.08% |
| **CLIP + DINO + VIT** | `geom` | 34.60% | 36.34% | **30.07%** | 33.01% |
| **CLIP + DINO + RESNET** | `mean` | 34.63% | 32.52% | **29.76%** | 31.85% |
| **CLIP + DINO + XCEPTION** | `mean` | 34.91% | 32.61% | **29.73%** | 32.43% |
| **CLIP + DINO + RESNET** | `geom` | 34.35% | 33.10% | **29.27%** | 31.27% |
| **CLIP + DINO + XCEPTION** | `geom` | 34.56% | 33.22% | **29.12%** | 30.69% |
| **TODOS OS 6 ROBUSTOS** | `mean` | 32.71% | 31.12% | **28.11%** | 31.08% |
| **TODOS OS 6 ROBUSTOS** | `geom` | 32.49% | 32.79% | **27.72%** | 28.19% |

## 3. Generalização no Benchmark DF40 por Paradigma Gerativo

| Categoria | Ensemble | Estratégia de Fusão | Modelos Integrados | AUC (%) | ACC (%) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123)** | `geom` | 3 | **86.44%** | 78.60% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123)** | `mean` | 3 | **86.41%** | 78.52% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123) + DINO Concat (s123)** | `mean` | 4 | **86.15%** | 78.50% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123) + DINO Concat (s123)** | `geom` | 4 | **85.58%** | 75.72% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + RESNET Concat (s123)** | `mean` | 2 | **85.42%** | 76.20% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + RESNET Concat (s123)** | `geom` | 2 | **85.40%** | 78.90% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + DINO Concat (s123) + RESNET Concat (s123)** | `mean` | 3 | **85.20%** | 77.68% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + RESNET Concat + DINO Concat + XCEPTION Concat** | `mean` | 4 | **84.89%** | 76.96% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + DINO Concat (s123) + RESNET Concat (s123)** | `geom` | 3 | **84.60%** | 75.72% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + DINO Robusto + RESNET Concat (s123)** | `mean` | 3 | **84.29%** | 76.31% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + DINO Robusto + RESNET Concat (s123)** | `geom` | 3 | **84.22%** | 73.45% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + RESNET Concat + DINO Concat + XCEPTION Concat** | `geom` | 4 | **84.11%** | 74.49% |
| Super-Ensemble Campeões (Melhores Modelos) | **SUPER-ENSEMBLE TOP-5 (CLIP Robusto + DINO Robusto + RESNET Concat + DINO Concat + XCEPTION Concat)** | `mean` | 5 | **84.05%** | 75.87% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + DINO Concat (s123) + RESNET Concat (s123)** | `max` | 3 | **83.78%** | 70.63% |
| Super-Ensemble Campeões (Melhores Modelos) | **SUPER-ENSEMBLE COMPLETO (Melhor de Cada Família)** | `geom` | 6 | **83.53%** | 72.76% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + RESNET Concat (s123)** | `max` | 2 | **83.40%** | 72.02% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123) + DINO Concat (s123)** | `max` | 4 | **83.37%** | 67.50% |
| Super-Ensemble Campeões (Melhores Modelos) | **CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123)** | `max` | 3 | **83.37%** | 68.50% |
| Super-Ensemble Campeões (Melhores Modelos) | **SUPER-ENSEMBLE TOP-5 (CLIP Robusto + DINO Robusto + RESNET Concat + DINO Concat + XCEPTION Concat)** | `geom` | 5 | **83.36%** | 72.61% |
| Super-Ensemble Campeões (Melhores Modelos) | **SUPER-ENSEMBLE COMPLETO (Melhor de Cada Família)** | `mean` | 6 | **83.33%** | 76.36% |
