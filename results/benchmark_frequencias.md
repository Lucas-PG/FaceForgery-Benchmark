# Benchmark de Modelos por Modo de Frequência (Decomposições FFT 2D)

Este documento reúne a análise de desempenho segmentada por cada uma das 7 formulações de entrada espectral investigadas no projeto.

> [!NOTE]
> Para a visão comparativa oficial consolidada de todos os 42 pares modelo-frequência ordenados por desempenho, consulte [`results/mostrar_rayson/tabela1-resultados-modelos-finetune.md`](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela1-resultados-modelos-finetune.md).

---

## Modo: RGB Espacial (none)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | 3 canais | 0.9173 ± 0.0058 | **0.7651 ± 0.0373** | -0.1522 | 0.7639 ± 0.0460 | 0.3324 ± 0.0310 | 0.2835 ± 0.0345 |
| **DINO (ConvNeXt-B)** | 3 canais | 0.9339 ± 0.0066 | **0.7370 ± 0.0506** | -0.1969 | 0.7208 ± 0.0289 | 0.3355 ± 0.0600 | 0.2863 ± 0.0749 |
| **Vision Transformer (ViT-B/16)** | 3 canais | 0.8149 ± 0.0078 | **0.7036 ± 0.0259** | -0.1112 | 0.7040 ± 0.0157 | 0.3521 ± 0.0370 | 0.3062 ± 0.0425 |
| **MobileNetV3-Large** | 3 canais | 0.8454 ± 0.0041 | **0.6832 ± 0.0299** | -0.1622 | 0.7037 ± 0.0140 | 0.3804 ± 0.0115 | 0.3297 ± 0.0179 |
| **ResNet-18** | 3 canais | 0.8765 ± 0.0201 | **0.6804 ± 0.0357** | -0.1961 | 0.6909 ± 0.0319 | 0.3352 ± 0.0278 | 0.2804 ± 0.0369 |
| **Xception** | 3 canais | 0.7708 ± 0.0058 | **0.6388 ± 0.0202** | -0.1321 | 0.7298 ± 0.0244 | 0.3979 ± 0.0144 | 0.3445 ± 0.0226 |

---

## Modo: Magnitude FFT (magnitude)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | 1 canais | 0.7177 ± 0.0050 | **0.5420 ± 0.0025** | -0.1757 | 0.4508 ± 0.0185 | 0.4602 ± 0.0063 | 0.4565 ± 0.0140 |
| **ResNet-18** | 1 canais | 0.7503 ± 0.0083 | **0.5409 ± 0.0118** | -0.2094 | 0.4510 ± 0.0493 | 0.4737 ± 0.0174 | 0.4725 ± 0.0298 |
| **DINO (ConvNeXt-B)** | 1 canais | 0.7614 ± 0.0028 | **0.5365 ± 0.0036** | -0.2249 | 0.4845 ± 0.0264 | 0.4890 ± 0.0105 | 0.4850 ± 0.0226 |
| **Vision Transformer (ViT-B/16)** | 1 canais | 0.7240 ± 0.0082 | **0.5347 ± 0.0080** | -0.1893 | 0.4447 ± 0.0197 | 0.4694 ± 0.0179 | 0.4494 ± 0.0264 |
| **Xception** | 1 canais | 0.7035 ± 0.0068 | **0.5328 ± 0.0071** | -0.1707 | 0.4180 ± 0.0267 | 0.4856 ± 0.0134 | 0.4760 ± 0.0210 |
| **CLIP ViT-B/16** | 1 canais | 0.7336 ± 0.0012 | **0.5300 ± 0.0037** | -0.2036 | 0.4827 ± 0.0245 | 0.4846 ± 0.0117 | 0.4822 ± 0.0159 |

---

## Modo: Fase FFT (phase)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | 1 canais | 0.7128 ± 0.0066 | **0.5332 ± 0.0081** | -0.1795 | 0.5206 ± 0.0257 | 0.4232 ± 0.0135 | 0.3723 ± 0.0169 |
| **MobileNetV3-Large** | 1 canais | 0.6287 ± 0.0042 | **0.5234 ± 0.0031** | -0.1052 | 0.5096 ± 0.0188 | 0.4685 ± 0.0045 | 0.4351 ± 0.0093 |
| **ResNet-18** | 1 canais | 0.6467 ± 0.0064 | **0.5211 ± 0.0021** | -0.1256 | 0.5052 ± 0.0120 | 0.4281 ± 0.0036 | 0.3547 ± 0.0105 |
| **Xception** | 1 canais | 0.5889 ± 0.0057 | **0.5111 ± 0.0045** | -0.0778 | 0.4807 ± 0.0255 | 0.4618 ± 0.0093 | 0.4325 ± 0.0130 |
| **CLIP ViT-B/16** | 1 canais | 0.6265 ± 0.0089 | **0.4994 ± 0.0020** | -0.1272 | 0.5180 ± 0.0330 | 0.4586 ± 0.0113 | 0.4422 ± 0.0202 |
| **Vision Transformer (ViT-B/16)** | 1 canais | 0.6020 ± 0.0046 | **0.4955 ± 0.0030** | -0.1064 | 0.5264 ± 0.0181 | 0.4734 ± 0.0129 | 0.4646 ± 0.0179 |

---

## Modo: Complexo Real+Imag (complex)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision Transformer (ViT-B/16)** | 2 canais | 0.5701 ± 0.0078 | **0.5541 ± 0.0051** | -0.0160 | 0.5093 ± 0.0110 | 0.4998 ± 0.0128 | 0.4982 ± 0.0187 |
| **ResNet-18** | 2 canais | 0.6992 ± 0.0311 | **0.5436 ± 0.0085** | -0.1556 | 0.4763 ± 0.0484 | 0.4330 ± 0.0433 | 0.4113 ± 0.0603 |
| **MobileNetV3-Large** | 2 canais | 0.6872 ± 0.0037 | **0.5337 ± 0.0027** | -0.1534 | 0.5982 ± 0.0450 | 0.4649 ± 0.0074 | 0.4408 ± 0.0137 |
| **CLIP ViT-B/16** | 2 canais | 0.5271 ± 0.0338 | **0.5197 ± 0.0226** | -0.0074 | 0.4723 ± 0.0548 | 0.5293 ± 0.0537 | 0.5299 ± 0.0596 |
| **Xception** | 2 canais | 0.5622 ± 0.0040 | **0.5174 ± 0.0037** | -0.0448 | 0.4895 ± 0.0140 | 0.4631 ± 0.0107 | 0.4372 ± 0.0148 |
| **DINO (ConvNeXt-B)** | 2 canais | 0.4995 ± 0.0012 | **0.4977 ± 0.0046** | -0.0018 | 0.5000 | 0.5025 ± 0.0054 | 0.5112 ± 0.0252 |

---

## Modo: RGB + Magnitude (concat)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | 4 canais | 0.8941 ± 0.0038 | **0.7260 ± 0.0106** | -0.1681 | 0.7650 ± 0.0323 | 0.3948 ± 0.0195 | 0.3648 ± 0.0254 |
| **Vision Transformer (ViT-B/16)** | 4 canais | 0.8265 ± 0.0031 | **0.7123 ± 0.0050** | -0.1142 | 0.6324 ± 0.0237 | 0.4519 ± 0.0146 | 0.4126 ± 0.0210 |
| **DINO (ConvNeXt-B)** | 4 canais | 0.9166 ± 0.0063 | **0.6834 ± 0.0091** | -0.2332 | 0.7558 ± 0.0207 | 0.3550 ± 0.0179 | 0.3059 ± 0.0213 |
| **ResNet-18** | 4 canais | 0.8717 ± 0.0102 | **0.6662 ± 0.0204** | -0.2055 | 0.7799 ± 0.0160 | 0.3853 ± 0.0310 | 0.3524 ± 0.0265 |
| **MobileNetV3-Large** | 4 canais | 0.8049 ± 0.0059 | **0.6296 ± 0.0023** | -0.1752 | 0.6676 ± 0.0177 | 0.4511 ± 0.0144 | 0.4294 ± 0.0179 |
| **Xception** | 4 canais | 0.7329 ± 0.0035 | **0.6119 ± 0.0023** | -0.1210 | 0.7344 ± 0.0108 | 0.4459 ± 0.0112 | 0.4191 ± 0.0191 |

---

## Modo: Filtros 3-Bandas (frequency_3)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | 1 canais | 0.7127 ± 0.0064 | **0.5452 ± 0.0079** | -0.1676 | 0.4258 ± 0.0305 | 0.4473 ± 0.0171 | 0.4305 ± 0.0195 |
| **DINO (ConvNeXt-B)** | 1 canais | 0.7631 ± 0.0148 | **0.5371 ± 0.0056** | -0.2260 | 0.5003 ± 0.0271 | 0.4655 ± 0.0231 | 0.4602 ± 0.0299 |
| **ResNet-18** | 1 canais | 0.7500 ± 0.0052 | **0.5365 ± 0.0047** | -0.2135 | 0.4421 ± 0.0104 | 0.4784 ± 0.0132 | 0.4750 ± 0.0178 |
| **Xception** | 1 canais | 0.6994 ± 0.0118 | **0.5330 ± 0.0098** | -0.1664 | 0.4558 ± 0.0317 | 0.4636 ± 0.0072 | 0.4408 ± 0.0084 |
| **CLIP ViT-B/16** | 1 canais | 0.7316 ± 0.0054 | **0.5328 ± 0.0062** | -0.1988 | 0.4725 ± 0.0271 | 0.4537 ± 0.0092 | 0.4256 ± 0.0160 |
| **Vision Transformer (ViT-B/16)** | 1 canais | 0.7240 ± 0.0065 | **0.5281 ± 0.0063** | -0.1959 | 0.4723 ± 0.0302 | 0.4587 ± 0.0150 | 0.4466 ± 0.0201 |

---

## Modo: Concat Espectral 7C (concat_frequency)

| Modelo | Canais | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 7 canais | 0.8337 ± 0.0065 | **0.6866 ± 0.0092** | -0.1472 | 0.7071 ± 0.0316 | 0.4344 ± 0.0231 | 0.3860 ± 0.0186 |
| **DINO (ConvNeXt-B)** | 7 canais | 0.8603 ± 0.0073 | **0.6820 ± 0.0090** | -0.1783 | 0.7053 ± 0.0391 | 0.4123 ± 0.0282 | 0.3664 ± 0.0192 |
| **Vision Transformer (ViT-B/16)** | 7 canais | 0.7915 ± 0.0065 | **0.6756 ± 0.0058** | -0.1159 | 0.5416 ± 0.0219 | 0.4893 ± 0.0033 | 0.4523 ± 0.0141 |
| **CLIP ViT-B/16** | 7 canais | 0.7789 ± 0.0064 | **0.6505 ± 0.0054** | -0.1284 | 0.5939 ± 0.0207 | 0.4690 ± 0.0185 | 0.4301 ± 0.0283 |
| **MobileNetV3-Large** | 7 canais | 0.7084 ± 0.0015 | **0.6032 ± 0.0027** | -0.1052 | 0.5635 ± 0.0248 | 0.4530 ± 0.0062 | 0.4147 ± 0.0220 |
| **Xception** | 7 canais | 0.6592 ± 0.0053 | **0.5746 ± 0.0049** | -0.0846 | 0.5973 ± 0.0158 | 0.4772 ± 0.0121 | 0.4578 ± 0.0206 |

---

## Síntese: Melhor Representação Espectral por Arquitetura

| Modelo | Melhor Modo no FF++ Limpo | Test AUC | Melhor Modo sob Perturbações (test_d) | Test-D AUC | Melhor Modo no DF40 | DF40 AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | `none` | 0.9173 | `none` | **0.7651** | `concat` | 0.7650 |
| **DINO (ConvNeXt-B)** | `none` | 0.9339 | `none` | **0.7370** | `concat` | 0.7558 |
| **Vision Transformer (ViT-B/16)** | `concat` | 0.8265 | `concat` | **0.7123** | `none` | 0.7040 |
| **ResNet-18** | `none` | 0.8765 | `concat_frequency` | **0.6866** | `concat` | 0.7799 |
| **MobileNetV3-Large** | `none` | 0.8454 | `none` | **0.6832** | `none` | 0.7037 |
| **Xception** | `none` | 0.7708 | `none` | **0.6388** | `concat` | 0.7344 |