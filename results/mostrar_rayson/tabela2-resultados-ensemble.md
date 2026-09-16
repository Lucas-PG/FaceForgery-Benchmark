# Tabela 2: Resultados Consolidados de Ensembles e Fusões Multimodais

**Documento:** `tabela2-resultados-ensemble.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Critério de Ordenação Estrito:** **Melhor ROC-AUC na Validação (`Val AUC` decrescente)**  

---

## 📌 Metodologia de Fusão e Estratégias Investigadas

As fusões combinam as predições probabilísticas $p_m(x) \in [0, 1]$ dos modelos através das seguintes abordagens:
1. **Média Simples (`mean`)**: $P(y=1|x) = \frac{1}{K} \sum_{m=1}^{K} p_m(x)$ (Soft Voting uniforme).
2. **Média Ponderada (`weighted`)**: $P(y=1|x) = \sum_{m=1}^{K} w_m \cdot p_m(x)$, onde $w_m \propto \text{AUC}_{\text{val}, m}$.
3. **Média Geométrica (`geometric`)**: $P(y=1|x) = \left( \prod_{m=1}^{K} p_m(x) \right)^{1/K}$ (Penaliza modelos discordantes com alta incerteza).
4. **Stacking / Regressão Logística (`stacking`)**: Meta-classificador linear $\sigma(W^T \mathbf{p} + b)$ treinado exclusivamente no conjunto de validação.
5. **Pesos Ótimos (`optimal_weights`)**: Otimização simplex SLSQP minimizando a perda logarítmica (Brier / Cross-Entropy) na validação.

---

## Tabela 2.2: Fusões de 2 Modelos ($K = 2$)

*(Total de 14 estratégias avaliadas — ordenado por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `geometric` | `RGB Robusto` | **0.9982** | 98.33% | 0.9517 | 88.94% | 89.69% | **0.8284** | -0.1233 | **0.8492** | 0.7812 |
| 2 | **CLIP+DINO Robustos** | `clip+dino` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9982** | 98.33% | 0.9400 | 85.94% | 86.77% | **0.8826** | -0.0574 | **0.8492** | 0.7812 |
| 3 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `mean` | `RGB Robusto` | **0.9981** | 98.58% | 0.9490 | 87.10% | 87.65% | **0.8234** | -0.1256 | **0.8492** | 0.7812 |
| 4 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `weighted` | `RGB Robusto` | **0.9981** | 98.58% | 0.9490 | 87.10% | 87.65% | **0.8234** | -0.1256 | **0.8492** | 0.7812 |
| 5 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `stacking` | `RGB Robusto` | **0.9981** | 98.65% | 0.9490 | 88.48% | 89.19% | **0.8105** | -0.1386 | **0.8492** | 0.7812 |
| 6 | **CLIP+DINO Robustos** | `clip+dino` | `mean` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.58% | 0.9380 | 84.61% | 85.07% | **0.8768** | -0.0612 | **0.8492** | 0.7812 |
| 7 | **CLIP+DINO Robustos** | `clip+dino` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.65% | 0.9379 | 85.48% | 86.11% | **0.8750** | -0.0629 | **0.8492** | 0.7812 |
| 8 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `max` | `RGB Robusto` | **0.9965** | 98.21% | 0.9441 | 87.79% | 88.55% | **0.7950** | -0.1492 | **0.8492** | 0.7812 |
| 9 | **CLIP+DINO Robustos** | `clip+dino` | `max` | `RGB Robusto Multi-Modelo` | **0.9965** | 98.21% | 0.9348 | 84.55% | 85.17% | **0.8711** | -0.0637 | **0.8492** | 0.7812 |
| 10 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `geometric` | `RGB Robusto` | **0.9875** | 94.38% | 0.9244 | 82.85% | 83.38% | **0.8243** | -0.1002 | **0.8263** | 0.7719 |
| 11 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `stacking` | `RGB Robusto` | **0.9871** | 94.47% | 0.9225 | 82.59% | 83.10% | **0.8266** | -0.0959 | **0.8263** | 0.7719 |
| 12 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `mean` | `RGB Robusto` | **0.9868** | 94.43% | 0.9240 | 83.14% | 83.72% | **0.8168** | -0.1072 | **0.8263** | 0.7719 |
| 13 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `weighted` | `RGB Robusto` | **0.9868** | 94.43% | 0.9240 | 83.14% | 83.72% | **0.8168** | -0.1072 | **0.8263** | 0.7719 |
| 14 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `max` | `RGB Robusto` | **0.9826** | 93.34% | 0.9211 | 84.19% | 85.15% | **0.7663** | -0.1548 | **0.8263** | 0.7719 |

---

## Tabela 2.3: Fusões de 3 Modelos ($K = 3$)

*(Total de 38 estratégias avaliadas — ordenado por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `stacking` | `Espacial RGB` | **0.9987** | 98.62% | 0.9590 | 88.60% | 89.31% | **0.7570** | -0.2020 | **0.8519** | 0.7012 |
| 2 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `stacking` | `Híbrido Espaço+Espectro` | **0.9986** | 98.66% | 0.9605 | 89.28% | 89.99% | **0.7705** | -0.1901 | **0.8654** | 0.7512 |
| 3 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `stacking` | `Híbrido RGB+Magnitude` | **0.9986** | 98.59% | 0.9597 | 88.82% | 89.50% | **0.7661** | -0.1936 | **0.8519** | 0.7012 |
| 4 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `optimal_weights` | `Espacial RGB` | **0.9985** | 97.86% | 0.9584 | 85.93% | 86.34% | **0.7511** | -0.2073 | **0.8519** | 0.7012 |
| 5 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `optimal_weights` | `Híbrido RGB+Magnitude` | **0.9985** | 97.86% | 0.9581 | 85.97% | 86.38% | **0.7521** | -0.2060 | **0.8519** | 0.7012 |
| 6 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `optimal_weights` | `Híbrido Espaço+Espectro` | **0.9985** | 97.86% | 0.9581 | 85.97% | 86.38% | **0.7521** | -0.2060 | **0.8654** | 0.7512 |
| 7 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9984** | 98.55% | 0.9336 | 84.05% | 84.63% | **0.8681** | -0.0656 | **0.8540** | 0.7850 |
| 8 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `geometric` | `Espacial RGB` | **0.9983** | 96.78% | 0.9553 | 81.68% | 81.31% | **0.7498** | -0.2055 | **0.8519** | 0.7012 |
| 9 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `weighted` | `Espacial RGB` | **0.9983** | 98.09% | 0.9551 | 86.39% | 86.91% | **0.7483** | -0.2068 | **0.8519** | 0.7012 |
| 10 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `mean` | `Espacial RGB` | **0.9983** | 98.07% | 0.9551 | 86.39% | 86.92% | **0.7483** | -0.2067 | **0.8519** | 0.7012 |
| 11 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `geometric` | `Híbrido Espaço+Espectro` | **0.9982** | 96.62% | 0.9601 | 84.11% | 84.17% | **0.7740** | -0.1861 | **0.8654** | 0.7512 |
| 12 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `geometric` | `Híbrido RGB+Magnitude` | **0.9981** | 97.07% | 0.9586 | 85.21% | 85.45% | **0.7667** | -0.1919 | **0.8519** | 0.7012 |
| 13 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `stacking` | `RGB Robusto` | **0.9981** | 98.63% | 0.9506 | 88.44% | 89.13% | **0.8000** | -0.1506 | **0.8492** | 0.7812 |
| 14 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.63% | 0.9340 | 85.76% | 86.55% | **0.8721** | -0.0619 | **0.8540** | 0.7850 |
| 15 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9980** | 98.15% | 0.9350 | 83.80% | 84.39% | **0.8755** | -0.0595 | **0.8540** | 0.7850 |
| 16 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9980** | 98.40% | 0.9323 | 85.41% | 86.15% | **0.8669** | -0.0654 | **0.8540** | 0.7850 |
| 17 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9978** | 98.10% | 0.9313 | 83.82% | 84.45% | **0.8753** | -0.0560 | **0.8540** | 0.7850 |
| 18 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `weighted` | `Híbrido Espaço+Espectro` | **0.9977** | 97.95% | 0.9578 | 89.07% | 89.80% | **0.7759** | -0.1819 | **0.8654** | 0.7512 |
| 19 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `mean` | `Híbrido Espaço+Espectro` | **0.9975** | 97.84% | 0.9570 | 89.03% | 89.78% | **0.7757** | -0.1813 | **0.8654** | 0.7512 |
| 20 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `geometric` | `RGB Robusto` | **0.9975** | 97.96% | 0.9511 | 87.89% | 88.54% | **0.8138** | -0.1373 | **0.8492** | 0.7812 |
| 21 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9975** | 97.80% | 0.9362 | 85.54% | 86.41% | **0.8779** | -0.0583 | **0.8540** | 0.7850 |
| 22 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `weighted` | `Híbrido RGB+Magnitude` | **0.9972** | 97.36% | 0.9565 | 88.66% | 89.42% | **0.7707** | -0.1858 | **0.8519** | 0.7012 |
| 23 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `mean` | `Híbrido RGB+Magnitude` | **0.9971** | 97.30% | 0.9562 | 88.64% | 89.40% | **0.7708** | -0.1855 | **0.8519** | 0.7012 |
| 24 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `weighted` | `RGB Robusto` | **0.9966** | 97.04% | 0.9480 | 86.04% | 86.58% | **0.8071** | -0.1408 | **0.8492** | 0.7812 |
| 25 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `mean` | `RGB Robusto` | **0.9966** | 97.04% | 0.9480 | 86.04% | 86.58% | **0.8071** | -0.1408 | **0.8492** | 0.7812 |
| 26 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `mean` | `RGB Robusto Multi-Modelo` | **0.9966** | 97.04% | 0.9255 | 85.20% | 86.13% | **0.8651** | -0.0605 | **0.8540** | 0.7850 |
| 27 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `mean` | `RGB Robusto Multi-Modelo` | **0.9963** | 97.64% | 0.9266 | 82.82% | 83.35% | **0.8589** | -0.0677 | **0.8540** | 0.7850 |
| 28 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `max` | `Espacial RGB` | **0.9962** | 94.90% | 0.9487 | 88.54% | 90.11% | **0.7408** | -0.2079 | **0.8519** | 0.7012 |
| 29 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `mean` | `RGB Robusto Multi-Modelo` | **0.9960** | 97.20% | 0.9200 | 85.39% | 86.49% | **0.8497** | -0.0703 | **0.8540** | 0.7850 |
| 30 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `max` | `RGB Robusto` | **0.9918** | 96.63% | 0.9406 | 87.29% | 88.23% | **0.7647** | -0.1758 | **0.8492** | 0.7812 |
| 31 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `max` | `RGB Robusto Multi-Modelo` | **0.9918** | 96.63% | 0.9244 | 83.33% | 83.84% | **0.8606** | -0.0638 | **0.8540** | 0.7850 |
| 32 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `max` | `RGB Robusto Multi-Modelo` | **0.9915** | 96.50% | 0.9122 | 82.95% | 83.88% | **0.8381** | -0.0740 | **0.8540** | 0.7850 |
| 33 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `max` | `Híbrido RGB+Magnitude` | **0.9911** | 92.33% | 0.9493 | 88.04% | 89.84% | **0.7632** | -0.1861 | **0.8519** | 0.7012 |
| 34 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `max` | `RGB Robusto Multi-Modelo` | **0.9910** | 96.00% | 0.9111 | 83.39% | 84.18% | **0.8234** | -0.0877 | **0.8540** | 0.7850 |
| 35 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `max` | `Híbrido Espaço+Espectro` | **0.9900** | 88.56% | 0.9442 | 84.37% | 87.41% | **0.7568** | -0.1874 | **0.8654** | 0.7512 |
| 36 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `majority` | `Espacial RGB` | **0.9782** | 97.71% | 0.8720 | 85.87% | 86.39% | **0.6736** | -0.1984 | **0.8519** | 0.7012 |
| 37 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `majority` | `Híbrido Espaço+Espectro` | **0.9695** | 97.10% | 0.8898 | 88.12% | 88.92% | **0.6866** | -0.2032 | **0.8654** | 0.7512 |
| 38 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `majority` | `Híbrido RGB+Magnitude` | **0.9633** | 96.53% | 0.8890 | 88.05% | 88.87% | **0.6976** | -0.1914 | **0.8519** | 0.7012 |

---

## Tabela 2.4: Fusões de 4 Modelos ($K = 4$)

*(Total de 17 estratégias avaliadas — ordenado por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `stacking` | `Híbrido Espaço+Espectro` | **0.9987** | 98.71% | 0.9614 | 89.49% | 90.20% | **0.7727** | -0.1886 | **0.8654** | 0.7512 |
| 2 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `optimal_weights` | `Híbrido Espaço+Espectro` | **0.9985** | 97.86% | 0.9581 | 85.96% | 86.38% | **0.7520** | -0.2061 | **0.8654** | 0.7512 |
| 3 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `stacking` | `RGB Robusto` | **0.9981** | 98.53% | 0.9514 | 88.19% | 88.84% | **0.7979** | -0.1535 | **0.8492** | 0.7812 |
| 4 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `geometric` | `Híbrido Espaço+Espectro` | **0.9979** | 96.48% | 0.9593 | 83.79% | 83.79% | **0.7735** | -0.1859 | **0.8654** | 0.7512 |
| 5 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `weighted` | `Híbrido Espaço+Espectro` | **0.9972** | 97.53% | 0.9582 | 89.45% | 90.23% | **0.7799** | -0.1783 | **0.8654** | 0.7512 |
| 6 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `mean` | `Híbrido Espaço+Espectro` | **0.9970** | 97.45% | 0.9578 | 89.45% | 90.24% | **0.7800** | -0.1778 | **0.8654** | 0.7512 |
| 7 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `geometric` | `RGB Robusto` | **0.9966** | 97.30% | 0.9497 | 86.59% | 87.12% | **0.8051** | -0.1446 | **0.8492** | 0.7812 |
| 8 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `mean` | `RGB Robusto` | **0.9955** | 96.62% | 0.9468 | 86.90% | 87.64% | **0.7991** | -0.1477 | **0.8492** | 0.7812 |
| 9 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `weighted` | `RGB Robusto` | **0.9955** | 96.62% | 0.9468 | 86.90% | 87.64% | **0.7991** | -0.1477 | **0.8492** | 0.7812 |
| 10 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `stacking` | `RGB Robusto` | **0.9892** | 95.05% | 0.9302 | 83.95% | 84.56% | **0.8126** | -0.1176 | **0.8263** | 0.7719 |
| 11 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `geometric` | `RGB Robusto` | **0.9891** | 94.70% | 0.9322 | 83.99% | 84.56% | **0.8013** | -0.1309 | **0.8263** | 0.7719 |
| 12 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `max` | `RGB Robusto` | **0.9888** | 95.62% | 0.9387 | 87.35% | 88.47% | **0.7625** | -0.1762 | **0.8492** | 0.7812 |
| 13 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `mean` | `RGB Robusto` | **0.9886** | 94.72% | 0.9318 | 83.96% | 84.45% | **0.7957** | -0.1361 | **0.8263** | 0.7719 |
| 14 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `weighted` | `RGB Robusto` | **0.9886** | 94.72% | 0.9318 | 83.96% | 84.45% | **0.7957** | -0.1361 | **0.8263** | 0.7719 |
| 15 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `max` | `Híbrido Espaço+Espectro` | **0.9882** | 86.13% | 0.9456 | 82.91% | 86.51% | **0.7498** | -0.1959 | **0.8654** | 0.7512 |
| 16 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `max` | `RGB Robusto` | **0.9847** | 93.73% | 0.9294 | 85.28% | 86.33% | **0.7549** | -0.1745 | **0.8263** | 0.7719 |
| 17 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `majority` | `Híbrido Espaço+Espectro` | **0.9478** | 95.59% | 0.8957 | 89.48% | 90.65% | **0.6770** | -0.2188 | **0.8654** | 0.7512 |

---

## Tabela 2.5: Fusões de 5 Modelos ($K = 5$)

*(Total de 17 estratégias avaliadas — ordenado por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `stacking` | `Híbrido Espaço+Espectro` | **0.9987** | 98.69% | 0.9614 | 89.46% | 90.17% | **0.7734** | -0.1880 | **0.8654** | 0.7512 |
| 2 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `optimal_weights` | `Híbrido Espaço+Espectro` | **0.9985** | 97.86% | 0.9581 | 85.96% | 86.38% | **0.7520** | -0.2061 | **0.8654** | 0.7512 |
| 3 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `stacking` | `RGB Robusto` | **0.9984** | 98.55% | 0.9527 | 87.77% | 88.38% | **0.8139** | -0.1388 | **0.8492** | 0.7812 |
| 4 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `geometric` | `RGB Robusto` | **0.9975** | 98.20% | 0.9522 | 88.67% | 89.43% | **0.8080** | -0.1441 | **0.8492** | 0.7812 |
| 5 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `geometric` | `Híbrido Espaço+Espectro` | **0.9973** | 95.20% | 0.9551 | 80.89% | 80.29% | **0.7717** | -0.1834 | **0.8654** | 0.7512 |
| 6 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `weighted` | `Híbrido Espaço+Espectro` | **0.9966** | 97.30% | 0.9552 | 88.23% | 88.97% | **0.7796** | -0.1756 | **0.8654** | 0.7512 |
| 7 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `mean` | `Híbrido Espaço+Espectro` | **0.9964** | 97.19% | 0.9544 | 88.10% | 88.84% | **0.7794** | -0.1750 | **0.8654** | 0.7512 |
| 8 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `mean` | `RGB Robusto` | **0.9963** | 97.64% | 0.9442 | 87.01% | 87.68% | **0.7963** | -0.1479 | **0.8492** | 0.7812 |
| 9 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `weighted` | `RGB Robusto` | **0.9963** | 97.64% | 0.9442 | 87.01% | 87.68% | **0.7963** | -0.1479 | **0.8492** | 0.7812 |
| 10 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `max` | `Híbrido Espaço+Espectro` | **0.9881** | 84.68% | 0.9448 | 81.67% | 85.71% | **0.7502** | -0.1946 | **0.8654** | 0.7512 |
| 11 | **CLIP Padrão (5 seeds, sem robust)** | `clip/none/{42,123,2024,7,2025}` | `geometric` | `Espacial RGB` | **0.9865** | 94.18% | 0.9345 | 85.13% | 85.92% | **0.7654** | -0.1692 | **0.8263** | 0.7719 |
| 12 | **CLIP Padrão (5 seeds, sem robust)** | `clip/none/{42,123,2024,7,2025}` | `stacking` | `Espacial RGB` | **0.9862** | 94.26% | 0.9343 | 85.19% | 85.97% | **0.7651** | -0.1691 | **0.8263** | 0.7719 |
| 13 | **CLIP Padrão (5 seeds, sem robust)** | `clip/none/{42,123,2024,7,2025}` | `mean` | `Espacial RGB` | **0.9861** | 94.30% | 0.9342 | 85.07% | 85.83% | **0.7654** | -0.1687 | **0.8263** | 0.7719 |
| 14 | **CLIP Padrão (5 seeds, sem robust)** | `clip/none/{42,123,2024,7,2025}` | `weighted` | `Espacial RGB` | **0.9861** | 94.30% | 0.9342 | 85.07% | 85.83% | **0.7654** | -0.1687 | **0.8263** | 0.7719 |
| 15 | **CLIP Padrão (5 seeds, sem robust)** | `clip/none/{42,123,2024,7,2025}` | `max` | `Espacial RGB` | **0.9837** | 93.35% | 0.9303 | 85.59% | 86.85% | **0.7473** | -0.1830 | **0.8263** | 0.7719 |
| 16 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `majority` | `Híbrido Espaço+Espectro` | **0.9638** | 96.49% | 0.8821 | 87.26% | 88.03% | **0.6958** | -0.1863 | **0.8654** | 0.7512 |
| 17 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `max` | `RGB Robusto` | **0.9573** | 90.43% | 0.8915 | 82.68% | 85.24% | **0.7129** | -0.1786 | **0.8492** | 0.7812 |

---

## Tabela 2.6: Fusões de 6 Modelos ($K = 6$)

*(Total de 4 estratégias avaliadas — ordenado por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9988** | 98.72% | 0.9326 | 84.32% | 84.95% | **0.8665** | -0.0661 | **0.8610** | 0.7940 |
| 2 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.15% | 0.9221 | 83.05% | 83.85% | **0.8595** | -0.0626 | **0.8610** | 0.7940 |
| 3 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `mean` | `RGB Robusto Multi-Modelo` | **0.9976** | 97.90% | 0.9105 | 81.89% | 82.78% | **0.8351** | -0.0753 | **0.8610** | 0.7940 |
| 4 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `max` | `RGB Robusto Multi-Modelo` | **0.9912** | 93.50% | 0.9018 | 82.21% | 83.18% | **0.8184** | -0.0834 | **0.8610** | 0.7940 |

---

## Síntese Comparativa: Melhor Ensemble de Cada Tamanho

| Qtd ($K$) | Melhor Composição | Estratégia | Val AUC (Pico) | Test AUC (Limpo) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K = 2** | **CLIP-987 + DINO-42 (2M)** | `geometric` | **0.9982** | 0.9517 | **0.8284** | -0.1233 | **0.8492** | 0.7812 |
| **K = 3** | **Seleção Ótima no Val (3 modelos)** | `stacking` | **0.9987** | 0.9590 | **0.7570** | -0.2020 | **0.8519** | 0.7012 |
| **K = 4** | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `stacking` | **0.9987** | 0.9614 | **0.7727** | -0.1886 | **0.8654** | 0.7512 |
| **K = 5** | **Top-5 Completo** | `stacking` | **0.9987** | 0.9614 | **0.7734** | -0.1880 | **0.8654** | 0.7512 |
| **K = 6** | **Todos Robustos (6M)** | `stacking` | **0.9988** | 0.9326 | **0.8665** | -0.0661 | **0.8610** | 0.7940 |