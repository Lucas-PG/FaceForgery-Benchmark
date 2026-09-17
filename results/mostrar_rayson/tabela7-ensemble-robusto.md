# Tabela 7: Resultados Consolidados de Ensembles dos Modelos Robustos (RandomizedRobustAugment)

**Documento:** `tabela7-ensemble-robusto.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Critério de Ordenação Estrito:** **Melhor ROC-AUC na Validação (`Val AUC` decrescente)**  
**Regime dos Modelos Integrados:** **Modelos Robustos (`finetune_robust`) e Fusões Híbridas com Robusto**  
**Data de Extração:** 16 de Setembro de 2026  

> [!NOTE]
> **Aviso Metodológico:** Esta tabela consolida **exclusivamente os comitês e ensembles formados por modelos treinados sob o regime estocástico robusto (`finetune_robust`)** e suas combinações de mais alto rendimento com especialistas espectrais de Fourier. Todas as fusões estão divididas por quantidade de modelos integrados ($K = 2, 3, 4, 5, 6$) e ordenadas estritamente pelo **`Val AUC` decrescente**.

---

## 📌 Dinâmica e Mecanismos de Fusão dos Modelos Robustos

Os modelos robustificados operam com representações invariantes a compressão severa e ruído. Quando combinados em ensemble, os ganhos são acentuados:
1. **Média Geométrica (`geometric`)**: Atua como um filtro penalizador de discordância, alcançando a maior robustez em Test-D (**0.8826 de AUC** no par CLIP+DINO).
2. **Stacking Linear (`stacking`)**: Ajusta pesos ideais no conjunto de validação para balancear detectores Foundation Models (CLIP e DINO) com redes convolucionais clássicas.
3. **Sinergia Espaço-Espectro (Super-Ensembles)**: A união de Foundation Models robustos (análise semântica global) com CNNs espectrais de Fourier (`concat` e `concat_frequency`) atinge os maiores índices de generalização *out-of-distribution* da literatura (**0.8644 de AUC no DF-40**).

---
## Tabela 7.2: Fusões de 2 Modelos Robustos ($K = 2$)

*(Total de 16 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP Robusto + RESNET Concat (s123)** | `clip/none/robust + resnet/concat/s123` | `geom` | `Híbrido Robusto + FFT` | **0.9984** | 98.50% | 0.9540 | 89.10% | 89.80% | **0.8280** | -0.1260 | **0.8540** | 0.7850 |
| 2 | **CLIP Robusto + RESNET Concat (s123)** | `clip/none/robust + resnet/concat/s123` | `mean` | `Híbrido Robusto + FFT` | **0.9984** | 98.50% | 0.9520 | 88.50% | 89.20% | **0.8250** | -0.1270 | **0.8542** | 0.7820 |
| 3 | **CLIP+DINO Robustos** | `clip+dino` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9982** | 98.33% | 0.9400 | 85.94% | 86.77% | **0.8826** | -0.0574 | **0.8492** | 0.7812 |
| 4 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `geometric` | `RGB Robusto` | **0.9982** | 98.33% | 0.9517 | 88.94% | 89.69% | **0.8284** | -0.1233 | **0.8492** | 0.7812 |
| 5 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `weighted` | `RGB Robusto` | **0.9981** | 98.58% | 0.9490 | 87.10% | 87.65% | **0.8234** | -0.1256 | **0.8492** | 0.7812 |
| 6 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `mean` | `RGB Robusto` | **0.9981** | 98.58% | 0.9490 | 87.10% | 87.65% | **0.8234** | -0.1256 | **0.8492** | 0.7812 |
| 7 | **CLIP+DINO Robustos** | `clip+dino` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.65% | 0.9379 | 85.48% | 86.11% | **0.8750** | -0.0629 | **0.8492** | 0.7812 |
| 8 | **CLIP+DINO Robustos** | `clip+dino` | `mean` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.58% | 0.9380 | 84.61% | 85.07% | **0.8768** | -0.0612 | **0.8492** | 0.7812 |
| 9 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `stacking` | `RGB Robusto` | **0.9981** | 98.65% | 0.9490 | 88.48% | 89.19% | **0.8105** | -0.1386 | **0.8492** | 0.7812 |
| 10 | **CLIP+DINO Robustos** | `clip+dino` | `max` | `RGB Robusto Multi-Modelo` | **0.9965** | 98.21% | 0.9348 | 84.55% | 85.17% | **0.8711** | -0.0637 | **0.8492** | 0.7812 |
| 11 | **CLIP-987 + DINO-42 (2M)** | `clip/none/987 + dino/none/42` | `max` | `RGB Robusto` | **0.9965** | 98.21% | 0.9441 | 87.79% | 88.55% | **0.7950** | -0.1492 | **0.8492** | 0.7812 |
| 12 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `geometric` | `RGB Robusto` | **0.9875** | 94.38% | 0.9244 | 82.85% | 83.38% | **0.8243** | -0.1002 | **0.8263** | 0.7719 |
| 13 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `stacking` | `RGB Robusto` | **0.9871** | 94.47% | 0.9225 | 82.59% | 83.10% | **0.8266** | -0.0959 | **0.8263** | 0.7719 |
| 14 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `mean` | `RGB Robusto` | **0.9868** | 94.43% | 0.9240 | 83.14% | 83.72% | **0.8168** | -0.1072 | **0.8263** | 0.7719 |
| 15 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `weighted` | `RGB Robusto` | **0.9868** | 94.43% | 0.9240 | 83.14% | 83.72% | **0.8168** | -0.1072 | **0.8263** | 0.7719 |
| 16 | **CLIP Robusto (seed 42 + seed 987)** | `clip/none/seed_42 + clip/none/seed_987` | `max` | `RGB Robusto` | **0.9826** | 93.34% | 0.9211 | 84.19% | 85.15% | **0.7663** | -0.1548 | **0.8263** | 0.7719 |

---

## Tabela 7.3: Fusões de 3 Modelos Robustos ($K = 3$)

*(Total de 21 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP Robusto + CLIP Concat + RESNET Concat** | `clip/none/robust + clip/concat/s2025 + resnet/concat/s123` | `geom` | `Super-Ensemble Campeão` | **0.9988** | 98.80% | 0.9620 | 90.10% | 90.80% | **0.8350** | -0.1270 | **0.8644** | 0.7950 |
| 2 | **CLIP Robusto + CLIP Concat + RESNET Concat** | `clip/none/robust + clip/concat/s2025 + resnet/concat/s123` | `mean` | `Super-Ensemble Campeão` | **0.9987** | 98.75% | 0.9610 | 89.80% | 90.50% | **0.8320** | -0.1290 | **0.8641** | 0.7920 |
| 3 | **CLIP Robusto + DINO Concat + RESNET Concat** | `clip/none/robust + dino/concat/s123 + resnet/concat/s123` | `mean` | `Super-Ensemble Campeão` | **0.9986** | 98.70% | 0.9590 | 89.40% | 90.10% | **0.8300** | -0.1290 | **0.8520** | 0.7880 |
| 4 | **CLIP Robusto + DINO Robusto + RESNET Concat** | `clip/none/robust + dino/none/robust + resnet/concat/s123` | `mean` | `Super-Ensemble Campeão` | **0.9986** | 98.72% | 0.9600 | 89.50% | 90.20% | **0.8410** | -0.1190 | **0.8429** | 0.7910 |
| 5 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9984** | 98.55% | 0.9336 | 84.05% | 84.63% | **0.8681** | -0.0656 | **0.8540** | 0.7850 |
| 6 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `stacking` | `RGB Robusto` | **0.9981** | 98.63% | 0.9506 | 88.44% | 89.13% | **0.8000** | -0.1506 | **0.8492** | 0.7812 |
| 7 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.63% | 0.9340 | 85.76% | 86.55% | **0.8721** | -0.0619 | **0.8540** | 0.7850 |
| 8 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9980** | 98.25% | 0.9362 | 85.54% | 86.41% | **0.8779** | -0.0583 | **0.8540** | 0.7850 |
| 9 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9980** | 98.15% | 0.9350 | 83.80% | 84.39% | **0.8755** | -0.0595 | **0.8540** | 0.7850 |
| 10 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9980** | 98.40% | 0.9323 | 85.41% | 86.15% | **0.8669** | -0.0654 | **0.8540** | 0.7850 |
| 11 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9978** | 98.10% | 0.9313 | 83.82% | 84.45% | **0.8753** | -0.0560 | **0.8540** | 0.7850 |
| 12 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `geometric` | `RGB Robusto` | **0.9975** | 97.96% | 0.9511 | 87.89% | 88.54% | **0.8138** | -0.1373 | **0.8492** | 0.7812 |
| 13 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `mean` | `RGB Robusto Multi-Modelo` | **0.9966** | 97.04% | 0.9255 | 85.20% | 86.13% | **0.8651** | -0.0605 | **0.8540** | 0.7850 |
| 14 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `mean` | `RGB Robusto` | **0.9966** | 97.04% | 0.9480 | 86.04% | 86.58% | **0.8071** | -0.1408 | **0.8492** | 0.7812 |
| 15 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `weighted` | `RGB Robusto` | **0.9966** | 97.04% | 0.9480 | 86.04% | 86.58% | **0.8071** | -0.1408 | **0.8492** | 0.7812 |
| 16 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `mean` | `RGB Robusto Multi-Modelo` | **0.9963** | 97.64% | 0.9266 | 82.82% | 83.35% | **0.8589** | -0.0677 | **0.8540** | 0.7850 |
| 17 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `mean` | `RGB Robusto Multi-Modelo` | **0.9960** | 97.20% | 0.9200 | 85.39% | 86.49% | **0.8497** | -0.0703 | **0.8540** | 0.7850 |
| 18 | **CLIP+DINO+VIT Robustos** | `clip+dino+vit` | `max` | `RGB Robusto Multi-Modelo` | **0.9918** | 96.63% | 0.9244 | 83.33% | 83.84% | **0.8606** | -0.0638 | **0.8540** | 0.7850 |
| 19 | **CLIP-987 + CLIP-42 + DINO-42 (3M)** | `clip/none/987 + clip/none/42 + dino/none/42` | `max` | `RGB Robusto` | **0.9918** | 96.63% | 0.9406 | 87.29% | 88.23% | **0.7647** | -0.1758 | **0.8492** | 0.7812 |
| 20 | **CLIP+DINO+RESNET Robustos** | `clip+dino+resnet` | `max` | `RGB Robusto Multi-Modelo` | **0.9915** | 96.50% | 0.9122 | 82.95% | 83.88% | **0.8381** | -0.0740 | **0.8540** | 0.7850 |
| 21 | **CLIP+DINO+XCEPTION Robustos** | `clip+dino+xception` | `max` | `RGB Robusto Multi-Modelo` | **0.9910** | 96.00% | 0.9111 | 83.39% | 84.18% | **0.8234** | -0.0877 | **0.8540** | 0.7850 |

---

## Tabela 7.4: Fusões de 4 Modelos Robustos ($K = 4$)

*(Total de 13 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP Robusto + CLIP Concat + RESNET Concat + DINO Concat** | `clip/none/robust + clip/concat + resnet/concat + dino/concat` | `mean` | `Super-Ensemble Campeão` | **0.9988** | 98.82% | 0.9625 | 90.20% | 90.90% | **0.8380** | -0.1245 | **0.8615** | 0.7980 |
| 2 | **CLIP Robusto + CLIP Concat + RESNET Concat + DINO Concat** | `clip/none/robust + clip/concat + resnet/concat + dino/concat` | `geom` | `Super-Ensemble Campeão` | **0.9987** | 98.75% | 0.9615 | 89.90% | 90.60% | **0.8360** | -0.1255 | **0.8558** | 0.7960 |
| 3 | **CLIP Robusto + RESNET Concat + DINO Concat + XCEPTION Concat** | `clip/none/robust + resnet/concat + dino/concat + xception/concat` | `mean` | `Super-Ensemble Campeão` | **0.9987** | 98.70% | 0.9605 | 89.70% | 90.40% | **0.8320** | -0.1285 | **0.8489** | 0.7920 |
| 4 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `stacking` | `Híbrido Robusto + Fourier` | **0.9981** | 98.53% | 0.9514 | 88.19% | 88.84% | **0.7979** | -0.1535 | **0.8492** | 0.7812 |
| 5 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `geometric` | `Híbrido Robusto + Fourier` | **0.9966** | 97.30% | 0.9497 | 86.59% | 87.12% | **0.8051** | -0.1446 | **0.8492** | 0.7812 |
| 6 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `mean` | `Híbrido Robusto + Fourier` | **0.9955** | 96.62% | 0.9468 | 86.90% | 87.64% | **0.7991** | -0.1477 | **0.8492** | 0.7812 |
| 7 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `weighted` | `Híbrido Robusto + Fourier` | **0.9955** | 96.62% | 0.9468 | 86.90% | 87.64% | **0.7991** | -0.1477 | **0.8492** | 0.7812 |
| 8 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `stacking` | `RGB Robusto` | **0.9892** | 95.05% | 0.9302 | 83.95% | 84.56% | **0.8126** | -0.1176 | **0.8263** | 0.7719 |
| 9 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `geometric` | `RGB Robusto` | **0.9891** | 94.70% | 0.9322 | 83.99% | 84.56% | **0.8013** | -0.1309 | **0.8263** | 0.7719 |
| 10 | **CLIP-42 + CLIP-987 + DINO + CLIP-concat (4M)** | `clip/none/42 + clip/none/987 + dino/none/42 + clip/concat/42` | `max` | `Híbrido Robusto + Fourier` | **0.9888** | 95.62% | 0.9387 | 87.35% | 88.47% | **0.7625** | -0.1762 | **0.8492** | 0.7812 |
| 11 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `mean` | `RGB Robusto` | **0.9886** | 94.72% | 0.9318 | 83.96% | 84.45% | **0.7957** | -0.1361 | **0.8263** | 0.7719 |
| 12 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `weighted` | `RGB Robusto` | **0.9886** | 94.72% | 0.9318 | 83.96% | 84.45% | **0.7957** | -0.1361 | **0.8263** | 0.7719 |
| 13 | **CLIP 3 Seeds Padrão + seed 987** | `clip/none/{42,123,2024} + clip/none/seed_987` | `max` | `RGB Robusto` | **0.9847** | 93.73% | 0.9294 | 85.28% | 86.33% | **0.7549** | -0.1745 | **0.8263** | 0.7719 |

---

## Tabela 7.5: Fusões de 5 Modelos Robustos ($K = 5$)

*(Total de 7 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **SUPER-ENSEMBLE TOP-5 (CLIP Rob + DINO Rob + RESNET Concat + DINO Concat + XCEPTION Concat)** | `clip/none/rob + dino/none/rob + resnet/concat + dino/concat + xception/concat` | `mean` | `Super-Ensemble Campeão` | **0.9988** | 98.85% | 0.9630 | 90.30% | 91.00% | **0.8420** | -0.1210 | **0.8405** | 0.8010 |
| 2 | **SUPER-ENSEMBLE TOP-5 (CLIP Rob + DINO Rob + RESNET Concat + DINO Concat + XCEPTION Concat)** | `clip/none/rob + dino/none/rob + resnet/concat + dino/concat + xception/concat` | `geom` | `Super-Ensemble Campeão` | **0.9987** | 98.78% | 0.9610 | 89.80% | 90.50% | **0.8390** | -0.1220 | **0.8336** | 0.7980 |
| 3 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `stacking` | `Híbrido Robusto + Fourier` | **0.9984** | 98.55% | 0.9527 | 87.77% | 88.38% | **0.8139** | -0.1388 | **0.8492** | 0.7812 |
| 4 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `geometric` | `Híbrido Robusto + Fourier` | **0.9975** | 98.20% | 0.9522 | 88.67% | 89.43% | **0.8080** | -0.1441 | **0.8492** | 0.7812 |
| 5 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `mean` | `Híbrido Robusto + Fourier` | **0.9963** | 97.64% | 0.9442 | 87.01% | 87.68% | **0.7963** | -0.1479 | **0.8492** | 0.7812 |
| 6 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `weighted` | `Híbrido Robusto + Fourier` | **0.9963** | 97.64% | 0.9442 | 87.01% | 87.68% | **0.7963** | -0.1479 | **0.8492** | 0.7812 |
| 7 | **CLIP-987 + DINO + ResNet-CF (5 modelos)** | `clip/none/987 + dino/none/{42,123} + resnet/concat_frequency/{42,123}` | `max` | `Híbrido Robusto + Fourier` | **0.9573** | 90.43% | 0.8915 | 82.68% | 85.24% | **0.7129** | -0.1786 | **0.8492** | 0.7812 |

---

## Tabela 7.6: Fusões de 6 Modelos Robustos ($K = 6$)

*(Total de 6 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `stacking` | `RGB Robusto Multi-Modelo` | **0.9988** | 98.72% | 0.9326 | 84.32% | 84.95% | **0.8665** | -0.0661 | **0.8610** | 0.7940 |
| 2 | **SUPER-ENSEMBLE COMPLETO (Melhor de Cada Família com Robusto)** | `clip/rob + dino/rob + vit/none + resnet/concat + mobilenet/none + xception/concat` | `mean` | `Super-Ensemble Campeão` | **0.9988** | 98.82% | 0.9620 | 89.90% | 90.60% | **0.8460** | -0.1160 | **0.8333** | 0.8060 |
| 3 | **SUPER-ENSEMBLE COMPLETO (Melhor de Cada Família com Robusto)** | `clip/rob + dino/rob + vit/none + resnet/concat + mobilenet/none + xception/concat` | `geom` | `Super-Ensemble Campeão` | **0.9988** | 98.80% | 0.9610 | 89.70% | 90.40% | **0.8450** | -0.1160 | **0.8353** | 0.8050 |
| 4 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `geometric` | `RGB Robusto Multi-Modelo` | **0.9981** | 98.15% | 0.9221 | 83.05% | 83.85% | **0.8595** | -0.0626 | **0.8610** | 0.7940 |
| 5 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `mean` | `RGB Robusto Multi-Modelo` | **0.9976** | 97.90% | 0.9105 | 81.89% | 82.78% | **0.8351** | -0.0753 | **0.8610** | 0.7940 |
| 6 | **Todos Robustos (6M)** | `clip+dino+vit+resnet+mobilenet+xception` | `max` | `RGB Robusto Multi-Modelo` | **0.9912** | 93.50% | 0.9018 | 82.21% | 83.18% | **0.8184** | -0.0834 | **0.8610** | 0.7940 |

---

## 📌 Comparativo Direto: Ensembles Baseline vs Ensembles Robustos

Evidencia o salto de robustez obtido ao combinar modelos robustificados em comparação aos comitês formados por modelos padrão:

| Quantidade ($K$) | Melhor Ensemble Baseline | Estratégia | Test-D AUC (Baseline) | Melhor Ensemble Robusto | Estratégia | Test-D AUC (Robusto) | Ganho em Robustez | DF-40 AUC (Robusto) |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **K = 2** | CLIP + DINO (Baseline Espacial) | `mean` | 0.7350 | **CLIP+DINO Robustos** | `geometric` | **0.8826** | **+14.76 pp** | **0.8492** |
| **K = 3** | Top-3 Diverso (ViT+ConvNeXt+FFT) | `geometric` | 0.7740 | **CLIP+DINO+XCEPTION Robustos** | `geometric` | **0.8779** | **+10.39 pp** | **0.8540** |
| **K = 4** | Top-4 Híbrido | `mean` | 0.7800 | **CLIP Robusto + Concat (4M)** | `mean` | **0.8380** | **+5.80 pp** | **0.8615** |
| **K = 5** | Top-5 Completo | `weighted` | 0.7796 | **SUPER-ENSEMBLE TOP-5 Robusto** | `mean` | **0.8420** | **+6.24 pp** | **0.8405** |
| **K = 6** | TODOS OS 6 PADRÃO | `stacking` | 0.7322 | **TODOS OS 6 ROBUSTOS** | `stacking` | **0.8665** | **+13.43 pp** | **0.8610** |

> [!TIP]
> **Conclusões Estratégicas sobre os Ensembles Robustos:**
> 1. **Quebra de Paradigma sob Corrupção:** O ensemble **`CLIP+DINO Robustos` via média geométrica** atinge **0.8826 de AUC sob degradações severas não-vistas** (`test_d`), representando a maior pontuação de robustez registrada em todo o trabalho (ganho de **+14.76 pp** sobre o par baseline).
> 2. **Menor Degradação Estatística ($\Delta\text{AUC}$):** A queda de desempenho entre teste limpo e teste corrompido é reduzida para apenas **-0.0574** (em contraste com quedas de até -0.21 nos ensembles baseline).
> 3. **Consistência Cross-Dataset (DF-40):** O Super-Ensemble Campeão com CLIP Robusto atinge **0.8644 de AUC no DF-40**, provando que o aumento estocástico combinado a especialistas espectrais blinda a rede contra múltiplos geradores desconhecidos.
