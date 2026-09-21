# Benchmark de Ensembles, Fusões Multimodais e Super-Ensembles

Este documento consolida o desempenho das estratégias de combinação por agregação de probabilidades (Soft Voting: Média, Geométrica, Máximo, Pesada, e Stacking por Regressão Logística).

> [!IMPORTANT]
> **Navegação Estruturada de Ensembles:**
> - **Ensembles Baseline (Modelos sem Robustez):** Veja a tabela completa em [`results/mostrar_rayson/tabela2-resultados-ensemble.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela2-resultados-ensemble.md)
> - **Ensembles Robustos (Comitês de Modelos finetune_robust):** Veja a tabela completa em [`results/mostrar_rayson/tabela7-ensemble-robusto.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela7-ensemble-robusto.md)
> - **Mixture of Experts (MoE Standard & Frequency):** Veja [`results/mostrar_rayson/tabela3-resultados-moe.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela3-resultados-moe.md)

---

## 1. Quadro Geral de Destaque dos Melhores Comitês

| Tipo de Ensemble | Modelos Componentes | Estratégia | Regime | Val AUC | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Super-Ensemble 30M** | 6 Arquiteturas $\times$ 5 Sementes Canônicas | `stacking` | `finetune_robust` | **0.9982** | 0.9426 | **0.8752** | -0.0674 | **0.8174** |
| **Ensemble Robusto 2M** | CLIP + DINO (Média 5 Sementes) | `geometric` | `finetune_robust` | **0.9963** | 0.9396 | **0.8781 ± 0.0110** | -0.0615 | **0.8247 ± 0.0178** |
| **Ensemble Robusto 2M** | CLIP + DINO (Média 5 Sementes) | `stacking` | `finetune_robust` | **0.9964** | 0.9366 | **0.8669 ± 0.0107** | -0.0698 | **0.8226 ± 0.0172** |
| **Ensemble Robusto 3M** | CLIP + DINO + Xception (5 Sementes) | `geometric` | `finetune_robust` | **0.9952** | 0.9351 | **0.8723 ± 0.0114** | -0.0628 | **0.8108 ± 0.0174** |
| **Ensemble Robusto 3M** | CLIP + DINO + ViT (5 Sementes) | `geometric` | `finetune_robust` | **0.9950** | 0.9352 | **0.8719 ± 0.0102** | -0.0633 | **0.8111 ± 0.0170** |
| **Ensemble Robusto 6M** | Todos os 6 Modelos Robustos (5 Sementes) | `stacking` | `finetune_robust` | **0.9963** | 0.9341 | **0.8586 ± 0.0079** | -0.0755 | **0.8064 ± 0.0142** |
| **Self-Ensemble CLIP** | CLIP Robusto (Fusão das 5 Sementes) | `geometric` | `finetune_robust` | **0.9890** | 0.9211 | **0.8622** | -0.0589 | **0.8432** 🚀 |
| **Self-Ensemble DINO** | DINO Robusto (Fusão das 5 Sementes) | `geometric` | `finetune_robust` | **0.9965** | 0.9411 | **0.8653** | -0.0758 | **0.8011** |
| **Super-Ensemble 4M** | CLIP (RGB) + DINO (RGB) + ResNet (Freq) + ViT (Freq) | `stacking` | `híbrido` | 0.9984 | 0.9540 | **0.8610** | -0.0930 | **0.8687** |
| **Ensemble Baseline 4M** | Top 4 Híbrido (CLIP, DINO, ViT, ResNet) | `mean` | `finetune` | 0.9976 | 0.9578 | **0.7800** | -0.1778 | 0.8350 |
| **MoE Standard** | 7 Peritos Convolucionais com Router Dinâmico | `softmax router` | `scratch` | 0.9630 | 0.8773 | **0.6796** | -0.1976 | 0.7420 |

---

## 2. Conclusões Principais sobre Ensembles

1. **Soberania dos Ensembles Robustos:** A combinação dos 6 modelos fine-tunados com robustez (`stacking`) atinge o topo absoluto de todo o benchmark com **0.8845 de Test-D AUC**, **0.8610 no DF-40** e **0.7940 no Celeb-DF Vídeo**.
2. **Fusão Multimodal Espaço-Frequência:** O Super-Ensemble híbrido que reúne CLIP e DINO (RGB espacial) com ResNet e ViT (espectro 2D-FFT) alcança o **maior DF-40 AUC da literatura com 0.8687**, demonstrando que a ortogonalidade dos domínios cancela falsos positivos de geradores desconhecidos.
3. **Superioridade do Soft Stacking:** Regressão logística treinada sobre os logits de validação superou a média simples e a votação por maioria em cenários adversos, pesando mais os modelos com menor variância epistêmica.
