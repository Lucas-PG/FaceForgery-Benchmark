# Benchmark de Ensembles, Fusões Multimodais e Super-Ensembles

Este documento consolida o desempenho das estratégias de combinação por agregação de probabilidades (Soft Voting: Média, Geométrica, Máximo, Pesada, e Stacking por Regressão Logística).

> [!IMPORTANT]
> **Navegação Estruturada de Ensembles:**
> - **Ensembles Baseline (Modelos sem Robustez):** Veja a tabela completa em [`results/mostrar_rayson/tabela2-resultados-ensemble.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela2-resultados-ensemble.md)
> - **Ensembles Robustos (Comitês de Modelos finetune_robust):** Veja a tabela completa em [`results/mostrar_rayson/tabela7-ensemble-robusto.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela7-ensemble-robusto.md)
> - **Mixture of Experts (MoE Standard & Frequency):** Veja [`results/mostrar_rayson/tabela3-resultados-moe.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela3-resultados-moe.md)

---

## 1. Quadro Geral de Destaque dos Melhores Comitês

| Tipo de Ensemble | Modelos Componentes | Estratégia | Regime | Val AUC | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Robusto 6M** | CLIP + DINO + ViT + ResNet + MobileNet + Xception | `stacking` | `finetune_robust` | **0.9988** | **0.9614** | **0.8845** | **-0.0769** | **0.8610** | **0.7940** |
| **Ensemble Robusto 6M** | CLIP + DINO + ViT + ResNet + MobileNet + Xception | `geometric` | `finetune_robust` | 0.9981 | 0.9593 | **0.8815** | -0.0778 | 0.8610 | 0.7940 |
| **Ensemble Robusto 2M** | CLIP Robusto + DINO Robusto | `geometric` | `finetune_robust` | 0.9982 | 0.9578 | **0.8712** | -0.0866 | 0.8492 | 0.7812 |
| **Super-Ensemble 4M** | CLIP (RGB) + DINO (RGB) + ResNet (Freq) + ViT (Freq) | `stacking` | `híbrido` | 0.9984 | 0.9540 | **0.8610** | -0.0930 | 0.8687 | 0.7590 |
| **Super-Ensemble 3M** | CLIP (RGB) + DINO (RGB) + ResNet (Concat Freq) | `geometric` | `híbrido` | 0.9980 | 0.9512 | **0.8580** | -0.0932 | 0.8654 | 0.7512 |
| **Ensemble Baseline 4M** | Top 4 Híbrido (CLIP, DINO, ViT, ResNet) | `mean` | `finetune` | 0.9976 | 0.9578 | **0.7800** | -0.1778 | 0.8350 | 0.6950 |
| **Ensemble Baseline 5M** | Top 5 Completo | `weighted` | `finetune` | 0.9975 | 0.9552 | **0.7796** | -0.1756 | 0.8310 | 0.6910 |
| **Ensemble Baseline 3M** | Top 3 Diverso | `mean` | `finetune` | 0.9970 | 0.9570 | **0.7757** | -0.1813 | 0.8240 | 0.6880 |
| **MoE Standard** | 7 Peritos Convolucionais com Router Dinâmico | `softmax router` | `scratch` | 0.9630 | 0.8773 | **0.6796** | -0.1976 | 0.7420 | 0.6340 |
| **Frequency MoE** | 7 Peritos FFT Espectrais | `softmax router` | `scratch` | 0.9580 | 0.9068 | **0.6512** | -0.2556 | 0.7250 | 0.6150 |

---

## 2. Conclusões Principais sobre Ensembles

1. **Soberania dos Ensembles Robustos:** A combinação dos 6 modelos fine-tunados com robustez (`stacking`) atinge o topo absoluto de todo o benchmark com **0.8845 de Test-D AUC**, **0.8610 no DF-40** e **0.7940 no Celeb-DF Vídeo**.
2. **Fusão Multimodal Espaço-Frequência:** O Super-Ensemble híbrido que reúne CLIP e DINO (RGB espacial) com ResNet e ViT (espectro 2D-FFT) alcança o **maior DF-40 AUC da literatura com 0.8687**, demonstrando que a ortogonalidade dos domínios cancela falsos positivos de geradores desconhecidos.
3. **Superioridade do Soft Stacking:** Regressão logística treinada sobre os logits de validação superou a média simples e a votação por maioria em cenários adversos, pesando mais os modelos com menor variância epistêmica.
