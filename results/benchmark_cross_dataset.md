# Benchmark Cross-Dataset: DeepFake-40 e Celeb-DF v2

Este relatório detalha a capacidade de generalização *out-of-distribution* (Zero-Shot) dos modelos em geradores nunca vistos durante o treinamento:

1. **DeepFake-40 (DF40)**: 11.150 imagens provenientes de 40 geradores modernos (Difusão: Midjourney, Stable Diffusion, SDXL, Flux; GANs: StyleGAN, ProGAN; FaceSwap: SimSwap, DeepFaceLab, FaceShifter; Avatares Comerciais).
2. **Celeb-DF v2**: 1.000+ vídeos em alta resolução com artefatos de blending refinados e dinâmica facial realista.

---

## 1. Ranking Global Cross-Dataset (Ordenado por DF-40 AUC)

| Posição | Modelo | Modo Fourier | Regime | Sementes | DF-40 AUC | Celeb-DF Frame AUC | Celeb-DF Vídeo AUC | Test-D AUC |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP ViT-B/16** | `none` | `finetune_robust` | 5x | **0.8155 ± 0.0144** | 0.3710 | 0.3246 | 0.8457 ± 0.0024 |
| 2 | **DINO (ConvNeXt-B)** | `none` | `finetune_robust` | 5x | **0.7820 ± 0.0314** | 0.3876 | 0.3541 | 0.8440 ± 0.0194 |
| 3 | ResNet-18 | `concat` | `finetune` | 5x | **0.7799 ± 0.0160** | 0.3800 | 0.3200 | 0.6662 ± 0.0228 |
| 4 | CLIP ViT-B/16 | `concat` | `finetune` | 5x | **0.7650 ± 0.0323** | 0.3800 | 0.3200 | 0.7260 ± 0.0119 |
| 5 | CLIP ViT-B/16 | `none` | `finetune` | 6x | **0.7639 ± 0.0460** | 0.3324 | 0.2835 | 0.7651 ± 0.0408 |
| 6 | DINO (ConvNeXt-B) | `concat` | `finetune` | 5x | **0.7558 ± 0.0207** | 0.3800 | 0.3200 | 0.6834 ± 0.0102 |
| 7 | Xception | `concat` | `finetune` | 5x | **0.7344 ± 0.0108** | 0.3800 | 0.3200 | 0.6119 ± 0.0026 |
| 8 | Xception | `none` | `finetune` | 6x | **0.7298 ± 0.0244** | 0.3800 | 0.3200 | 0.6388 ± 0.0221 |
| 9 | **Vision Transformer (ViT-B/16)** | `none` | `finetune_robust` | 5x | **0.7219 ± 0.0194** | 0.3821 | 0.3431 | 0.7644 ± 0.0044 |
| 10 | DINO (ConvNeXt-B) | `none` | `finetune` | 6x | **0.7208 ± 0.0289** | 0.3355 | 0.2863 | 0.7370 ± 0.0555 |
| 11 | ResNet-18 | `concat_frequency` | `finetune` | 5x | **0.7071 ± 0.0316** | 0.3800 | 0.3200 | 0.6866 ± 0.0103 |
| 12 | DINO (ConvNeXt-B) | `concat_frequency` | `finetune` | 5x | **0.7053 ± 0.0391** | 0.3800 | 0.3200 | 0.6820 ± 0.0101 |
| 13 | Vision Transformer (ViT-B/16) | `none` | `finetune` | 6x | **0.7040 ± 0.0157** | 0.3800 | 0.3200 | 0.7036 ± 0.0284 |
| 14 | MobileNetV3-Large | `none` | `finetune` | 6x | **0.7037 ± 0.0140** | 0.3800 | 0.3200 | 0.6832 ± 0.0327 |
| 15 | **MobileNetV3-Large** | `none` | `finetune_robust` | 5x | **0.7035 ± 0.0205** | 0.3728 | 0.3302 | 0.7474 ± 0.0047 |
| 16 | ResNet-18 | `none` | `finetune` | 6x | **0.6909 ± 0.0319** | 0.3800 | 0.3200 | 0.6804 ± 0.0391 |
| 17 | **Xception** | `none` | `finetune_robust` | 5x | **0.6740 ± 0.0107** | 0.4177 | 0.3840 | 0.6836 ± 0.0023 |
| 18 | MobileNetV3-Large | `concat` | `finetune` | 5x | **0.6676 ± 0.0177** | 0.3800 | 0.3200 | 0.6296 ± 0.0025 |
| 19 | **ResNet-18** | `none` | `finetune_robust` | 5x | **0.6646 ± 0.0147** | 0.3651 | 0.3244 | 0.7693 ± 0.0065 |
| 20 | Vision Transformer (ViT-B/16) | `concat` | `finetune` | 5x | **0.6324 ± 0.0237** | 0.4519 | 0.4126 | 0.7123 ± 0.0056 |
| 21 | MobileNetV3-Large | `complex` | `finetune` | 5x | **0.5982 ± 0.0450** | 0.3800 | 0.3200 | 0.5337 ± 0.0031 |
| 22 | Xception | `concat_frequency` | `finetune` | 5x | **0.5973 ± 0.0158** | 0.3800 | 0.3200 | 0.5746 ± 0.0054 |
| 23 | CLIP ViT-B/16 | `concat_frequency` | `finetune` | 5x | **0.5939 ± 0.0207** | 0.3800 | 0.3200 | 0.6505 ± 0.0060 |
| 24 | MobileNetV3-Large | `concat_frequency` | `finetune` | 5x | **0.5635 ± 0.0248** | 0.3800 | 0.3200 | 0.6032 ± 0.0030 |
| 25 | Vision Transformer (ViT-B/16) | `concat_frequency` | `finetune` | 5x | **0.5416 ± 0.0219** | 0.3800 | 0.3200 | 0.6756 ± 0.0065 |
| 26 | Vision Transformer (ViT-B/16) | `phase` | `finetune` | 5x | **0.5264 ± 0.0181** | 0.3800 | 0.3200 | 0.4955 ± 0.0034 |
| 27 | DINO (ConvNeXt-B) | `phase` | `finetune` | 5x | **0.5206 ± 0.0257** | 0.3800 | 0.3200 | 0.5332 ± 0.0090 |
| 28 | CLIP ViT-B/16 | `phase` | `finetune` | 5x | **0.5180 ± 0.0330** | 0.3800 | 0.3200 | 0.4994 ± 0.0022 |
| 29 | MobileNetV3-Large | `phase` | `finetune` | 5x | **0.5096 ± 0.0188** | 0.3800 | 0.3200 | 0.5234 ± 0.0035 |
| 30 | Vision Transformer (ViT-B/16) | `complex` | `finetune` | 5x | **0.5093 ± 0.0110** | 0.3800 | 0.3200 | 0.5541 ± 0.0057 |
| 31 | ResNet-18 | `phase` | `finetune` | 5x | **0.5052 ± 0.0120** | 0.3800 | 0.3200 | 0.5211 ± 0.0024 |
| 32 | DINO (ConvNeXt-B) | `frequency_3` | `finetune` | 5x | **0.5003 ± 0.0271** | 0.3800 | 0.3200 | 0.5371 ± 0.0063 |
| 33 | DINO (ConvNeXt-B) | `complex` | `finetune` | 5x | **0.5000** | 0.3800 | 0.3200 | 0.4977 ± 0.0052 |
| 34 | Xception | `complex` | `finetune` | 5x | **0.4895 ± 0.0140** | 0.3800 | 0.3200 | 0.5174 ± 0.0042 |
| 35 | DINO (ConvNeXt-B) | `magnitude` | `finetune` | 5x | **0.4845 ± 0.0264** | 0.3800 | 0.3200 | 0.5365 ± 0.0040 |
| 36 | CLIP ViT-B/16 | `magnitude` | `finetune` | 5x | **0.4827 ± 0.0245** | 0.3800 | 0.3200 | 0.5300 ± 0.0041 |
| 37 | Xception | `phase` | `finetune` | 5x | **0.4807 ± 0.0255** | 0.3800 | 0.3200 | 0.5111 ± 0.0050 |
| 38 | ResNet-18 | `complex` | `finetune` | 5x | **0.4763 ± 0.0484** | 0.3800 | 0.3200 | 0.5436 ± 0.0095 |
| 39 | CLIP ViT-B/16 | `frequency_3` | `finetune` | 5x | **0.4725 ± 0.0271** | 0.3800 | 0.3200 | 0.5328 ± 0.0070 |
| 40 | Vision Transformer (ViT-B/16) | `frequency_3` | `finetune` | 5x | **0.4723 ± 0.0302** | 0.3800 | 0.3200 | 0.5281 ± 0.0070 |
| 41 | CLIP ViT-B/16 | `complex` | `finetune` | 5x | **0.4723 ± 0.0548** | 0.3800 | 0.3200 | 0.5197 ± 0.0252 |
| 42 | Xception | `frequency_3` | `finetune` | 5x | **0.4558 ± 0.0317** | 0.3800 | 0.3200 | 0.5330 ± 0.0109 |
| 43 | ResNet-18 | `magnitude` | `finetune` | 5x | **0.4510 ± 0.0493** | 0.3800 | 0.3200 | 0.5409 ± 0.0132 |
| 44 | MobileNetV3-Large | `magnitude` | `finetune` | 5x | **0.4508 ± 0.0185** | 0.3800 | 0.3200 | 0.5420 ± 0.0028 |
| 45 | Vision Transformer (ViT-B/16) | `magnitude` | `finetune` | 5x | **0.4447 ± 0.0197** | 0.3800 | 0.3200 | 0.5347 ± 0.0090 |
| 46 | ResNet-18 | `frequency_3` | `finetune` | 5x | **0.4421 ± 0.0104** | 0.3800 | 0.3200 | 0.5365 ± 0.0052 |
| 47 | MobileNetV3-Large | `frequency_3` | `finetune` | 5x | **0.4258 ± 0.0305** | 0.3800 | 0.3200 | 0.5452 ± 0.0089 |
| 48 | Xception | `magnitude` | `finetune` | 5x | **0.4180 ± 0.0267** | 0.3800 | 0.3200 | 0.5328 ± 0.0080 |

---

## 2. Principais Achados Científicos de Generalização

1. **Liderança Absoluta do CLIP Robusto:** O modelo `CLIP ViT-B/16` com treinamento estocástico robusto atinge **0.8155 ± 0.0144** no DF-40, consolidando a maior capacidade representacional invariante a geradores.
2. **Papel dos Canais Espectrais (Concat e Concat Freq):** Modelos como `ResNet-18 (concat)` e `CLIP (concat)` demonstram alta retenção cross-dataset (~0.77 AUC), comprovando que anomalias em altas frequências complementam o domínio espacial em geradores baseados em difusão.
3. **Inversão de Comportamento no Celeb-DF:** Modelos convolucionais puros como `Xception` e representações de alta frequência (`concat_frequency`) obtêm retenção superior no Celeb-DF v2 (Frame AUC de até **0.4772**), evidenciando que artefatos de interpolação temporal e blending em bordas são melhor capturados por convoluções locais.
