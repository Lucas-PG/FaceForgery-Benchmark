# Benchmark Cross-Dataset: DeepFake-40 e Celeb-DF v2

Este relatório detalha a capacidade de generalização *out-of-distribution* dos modelos em geradores nunca vistos durante o treinamento:

1. **DeepFake-40 (DF40)**: 11.150 imagens de 40 geradores modernos (Difusão: Midjourney, Stable Diffusion, SDXL, Flux; GANs: StyleGAN, ProGAN; FaceSwap: SimSwap, DeepFaceLab, FaceShifter; Avatares Comerciais).

2. **Celeb-DF v2**: 1.000+ vídeos em alta resolução com artefatos de blending sutilmente refinados.

---

## 1. Ranking Global Cross-Dataset (DF40 e Celeb-DF v2)

| Posição | Modelo | Modo Fourier | Regime | DF-40 AUC | Celeb-DF Frame AUC | Celeb-DF Vídeo AUC | Test-D AUC |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **CLIP ViT-B/16** | `none` | `finetune_robust` | **0.8199 ± 0.0105** | 0.3710 | 0.3246 | 0.8462 ± 0.0021 |
| 2 | **ResNet-18** | `concat` | `finetune` | **0.7799 ± 0.0160** | 0.3853 ± 0.0310 | 0.3524 ± 0.0265 | 0.6662 ± 0.0204 |
| 3 | **CLIP ViT-B/16** | `concat` | `finetune` | **0.7650 ± 0.0323** | 0.3948 ± 0.0195 | 0.3648 ± 0.0254 | 0.7260 ± 0.0106 |
| 4 | **CLIP ViT-B/16** | `none` | `finetune` | **0.7639 ± 0.0460** | 0.3324 ± 0.0310 | 0.2835 ± 0.0345 | 0.7651 ± 0.0373 |
| 5 | **DINO (ConvNeXt-B)** | `concat` | `finetune` | **0.7558 ± 0.0207** | 0.3550 ± 0.0179 | 0.3059 ± 0.0213 | 0.6834 ± 0.0091 |
| 6 | **DINO (ConvNeXt-B)** | `none` | `finetune_robust` | **0.7517** | 0.3876 | 0.3541 | 0.8483 |
| 7 | **Xception** | `concat` | `finetune` | **0.7344 ± 0.0108** | 0.4459 ± 0.0112 | 0.4191 ± 0.0191 | 0.6119 ± 0.0023 |
| 8 | **MobileNetV3-Large** | `none` | `finetune_robust` | **0.7304** | 0.3728 | 0.3302 | 0.7493 |
| 9 | **Xception** | `none` | `finetune` | **0.7298 ± 0.0244** | 0.3979 ± 0.0144 | 0.3445 ± 0.0226 | 0.6388 ± 0.0202 |
| 10 | **Vision Transformer (ViT-B/16)** | `none` | `finetune_robust` | **0.7216 ± 0.0069** | 0.3821 | 0.3431 | 0.7621 ± 0.0021 |
| 11 | **DINO (ConvNeXt-B)** | `none` | `finetune` | **0.7208 ± 0.0289** | 0.3355 ± 0.0600 | 0.2863 ± 0.0749 | 0.7370 ± 0.0506 |
| 12 | **ResNet-18** | `concat_frequency` | `finetune` | **0.7071 ± 0.0316** | 0.4344 ± 0.0231 | 0.3860 ± 0.0186 | 0.6866 ± 0.0092 |
| 13 | **DINO (ConvNeXt-B)** | `concat_frequency` | `finetune` | **0.7053 ± 0.0391** | 0.4123 ± 0.0282 | 0.3664 ± 0.0192 | 0.6820 ± 0.0090 |
| 14 | **Vision Transformer (ViT-B/16)** | `none` | `finetune` | **0.7040 ± 0.0157** | 0.3521 ± 0.0370 | 0.3062 ± 0.0425 | 0.7036 ± 0.0259 |
| 15 | **MobileNetV3-Large** | `none` | `finetune` | **0.7037 ± 0.0140** | 0.3804 ± 0.0115 | 0.3297 ± 0.0179 | 0.6832 ± 0.0299 |
| 16 | **ResNet-18** | `none` | `finetune` | **0.6909 ± 0.0319** | 0.3352 ± 0.0278 | 0.2804 ± 0.0369 | 0.6804 ± 0.0357 |
| 17 | **Xception** | `none` | `finetune_robust` | **0.6814** | 0.4177 | 0.3840 | 0.6838 |
| 18 | **MobileNetV3-Large** | `concat` | `finetune` | **0.6676 ± 0.0177** | 0.4511 ± 0.0144 | 0.4294 ± 0.0179 | 0.6296 ± 0.0023 |
| 19 | **ResNet-18** | `none` | `finetune_robust` | **0.6451** | 0.3651 | 0.3244 | 0.7587 |
| 20 | **Vision Transformer (ViT-B/16)** | `concat` | `finetune` | **0.6324 ± 0.0237** | 0.4519 ± 0.0146 | 0.4126 ± 0.0210 | 0.7123 ± 0.0050 |
| 21 | **MobileNetV3-Large** | `complex` | `finetune` | **0.5982 ± 0.0450** | 0.4649 ± 0.0074 | 0.4408 ± 0.0137 | 0.5337 ± 0.0027 |
| 22 | **Xception** | `concat_frequency` | `finetune` | **0.5973 ± 0.0158** | 0.4772 ± 0.0121 | 0.4578 ± 0.0206 | 0.5746 ± 0.0049 |
| 23 | **CLIP ViT-B/16** | `concat_frequency` | `finetune` | **0.5939 ± 0.0207** | 0.4690 ± 0.0185 | 0.4301 ± 0.0283 | 0.6505 ± 0.0054 |
| 24 | **MobileNetV3-Large** | `concat_frequency` | `finetune` | **0.5635 ± 0.0248** | 0.4530 ± 0.0062 | 0.4147 ± 0.0220 | 0.6032 ± 0.0027 |
| 25 | **Vision Transformer (ViT-B/16)** | `concat_frequency` | `finetune` | **0.5416 ± 0.0219** | 0.4893 ± 0.0033 | 0.4523 ± 0.0141 | 0.6756 ± 0.0058 |
| 26 | **Vision Transformer (ViT-B/16)** | `phase` | `finetune` | **0.5264 ± 0.0181** | 0.4734 ± 0.0129 | 0.4646 ± 0.0179 | 0.4955 ± 0.0030 |
| 27 | **DINO (ConvNeXt-B)** | `phase` | `finetune` | **0.5206 ± 0.0257** | 0.4232 ± 0.0135 | 0.3723 ± 0.0169 | 0.5332 ± 0.0081 |
| 28 | **CLIP ViT-B/16** | `phase` | `finetune` | **0.5180 ± 0.0330** | 0.4586 ± 0.0113 | 0.4422 ± 0.0202 | 0.4994 ± 0.0020 |
| 29 | **MobileNetV3-Large** | `phase` | `finetune` | **0.5096 ± 0.0188** | 0.4685 ± 0.0045 | 0.4351 ± 0.0093 | 0.5234 ± 0.0031 |
| 30 | **Vision Transformer (ViT-B/16)** | `complex` | `finetune` | **0.5093 ± 0.0110** | 0.4998 ± 0.0128 | 0.4982 ± 0.0187 | 0.5541 ± 0.0051 |
| 31 | **ResNet-18** | `phase` | `finetune` | **0.5052 ± 0.0120** | 0.4281 ± 0.0036 | 0.3547 ± 0.0105 | 0.5211 ± 0.0021 |
| 32 | **DINO (ConvNeXt-B)** | `frequency_3` | `finetune` | **0.5003 ± 0.0271** | 0.4655 ± 0.0231 | 0.4602 ± 0.0299 | 0.5371 ± 0.0056 |
| 33 | **DINO (ConvNeXt-B)** | `complex` | `finetune` | **0.5000** | 0.5025 ± 0.0054 | 0.5112 ± 0.0252 | 0.4977 ± 0.0046 |
| 34 | **Xception** | `complex` | `finetune` | **0.4895 ± 0.0140** | 0.4631 ± 0.0107 | 0.4372 ± 0.0148 | 0.5174 ± 0.0037 |
| 35 | **DINO (ConvNeXt-B)** | `magnitude` | `finetune` | **0.4845 ± 0.0264** | 0.4890 ± 0.0105 | 0.4850 ± 0.0226 | 0.5365 ± 0.0036 |
| 36 | **CLIP ViT-B/16** | `magnitude` | `finetune` | **0.4827 ± 0.0245** | 0.4846 ± 0.0117 | 0.4822 ± 0.0159 | 0.5300 ± 0.0037 |
| 37 | **Xception** | `phase` | `finetune` | **0.4807 ± 0.0255** | 0.4618 ± 0.0093 | 0.4325 ± 0.0130 | 0.5111 ± 0.0045 |
| 38 | **ResNet-18** | `complex` | `finetune` | **0.4763 ± 0.0484** | 0.4330 ± 0.0433 | 0.4113 ± 0.0603 | 0.5436 ± 0.0085 |
| 39 | **CLIP ViT-B/16** | `frequency_3` | `finetune` | **0.4725 ± 0.0271** | 0.4537 ± 0.0092 | 0.4256 ± 0.0160 | 0.5328 ± 0.0062 |
| 40 | **Vision Transformer (ViT-B/16)** | `frequency_3` | `finetune` | **0.4723 ± 0.0302** | 0.4587 ± 0.0150 | 0.4466 ± 0.0201 | 0.5281 ± 0.0063 |
| 41 | **CLIP ViT-B/16** | `complex` | `finetune` | **0.4723 ± 0.0548** | 0.5293 ± 0.0537 | 0.5299 ± 0.0596 | 0.5197 ± 0.0226 |
| 42 | **Xception** | `frequency_3` | `finetune` | **0.4558 ± 0.0317** | 0.4636 ± 0.0072 | 0.4408 ± 0.0084 | 0.5330 ± 0.0098 |
| 43 | **ResNet-18** | `magnitude` | `finetune` | **0.4510 ± 0.0493** | 0.4737 ± 0.0174 | 0.4725 ± 0.0298 | 0.5409 ± 0.0118 |
| 44 | **MobileNetV3-Large** | `magnitude` | `finetune` | **0.4508 ± 0.0185** | 0.4602 ± 0.0063 | 0.4565 ± 0.0140 | 0.5420 ± 0.0025 |
| 45 | **Vision Transformer (ViT-B/16)** | `magnitude` | `finetune` | **0.4447 ± 0.0197** | 0.4694 ± 0.0179 | 0.4494 ± 0.0264 | 0.5347 ± 0.0080 |
| 46 | **ResNet-18** | `frequency_3` | `finetune` | **0.4421 ± 0.0104** | 0.4784 ± 0.0132 | 0.4750 ± 0.0178 | 0.5365 ± 0.0047 |
| 47 | **MobileNetV3-Large** | `frequency_3` | `finetune` | **0.4258 ± 0.0305** | 0.4473 ± 0.0171 | 0.4305 ± 0.0195 | 0.5452 ± 0.0079 |
| 48 | **Xception** | `magnitude` | `finetune` | **0.4180 ± 0.0267** | 0.4856 ± 0.0134 | 0.4760 ± 0.0210 | 0.5328 ± 0.0071 |