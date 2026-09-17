# Benchmark dos Modelos Robustos (RandomizedRobustAugment)

Este relatório detalha a avaliação dos 6 modelos fine-tunados com a pipeline estocástica `RandomizedRobustAugment` (ruído gaussiano, compressão JPEG agressiva, rotações, superexposição, contraste e blur):

- **Sementes Canônicas Definidas (5 seeds)**: `[987, 42, 123, 2024, 7]` *(aproveitando a seed 987 pré-concluída e eliminando a 2025)*
- **Hardware em Execução**: GPU 0 (`clip`, `dino`, `resnet`) | GPU 1 (`vit`, `mobilenet`, `xception`)

---

## 1. Tabela Comparativa: Baseline Limpo vs Treino Robusto

| Modelo | Regime | Seeds Concluídas | Status de Execução | Test AUC (Limpo) | Test-D AUC (Corrompido) | ΔAUC (Degradação) | DF-40 AUC (Cross-Gen) | Celeb-DF Frame |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CLIP ViT-B/16 | `finetune` (Baseline) | 6 seeds | ✅ Concluído | 0.9173 ± 0.0058 | 0.7651 ± 0.0373 | -0.1522 | 0.7639 ± 0.0460 | 0.3324 ± 0.0310 |
| **CLIP ViT-B/16** | `finetune_robust` | **4/5 seeds** | **🔄 Em andamento (seed 7 na GPU 0)** | **0.9077 ± 0.0014** | **0.8462 ± 0.0021** | **-0.0615** | **0.8199 ± 0.0105** | **0.3710** |
| DINO (ConvNeXt-B) | `finetune` (Baseline) | 6 seeds | ✅ Concluído | 0.9339 ± 0.0066 | 0.7370 ± 0.0506 | -0.1969 | 0.7208 ± 0.0289 | 0.3355 ± 0.0600 |
| **DINO (ConvNeXt-B)** | `finetune_robust` | **1/5 seeds** | **⏳ Na fila (GPU 0)** | **0.9242** | **0.8483** | **-0.0759** | **0.7517** | **0.3876** |
| Vision Transformer (ViT-B/16) | `finetune` (Baseline) | 6 seeds | ✅ Concluído | 0.8149 ± 0.0078 | 0.7036 ± 0.0259 | -0.1112 | 0.7040 ± 0.0157 | 0.3521 ± 0.0370 |
| **Vision Transformer (ViT-B/16)** | `finetune_robust` | **3/5 seeds** | **🔄 Em andamento (seed 2024 na GPU 1)** | **0.8210 ± 0.0032** | **0.7621 ± 0.0021** | **-0.0589** | **0.7216 ± 0.0069** | **0.3821** |
| ResNet-18 | `finetune` (Baseline) | 6 seeds | ✅ Concluído | 0.8765 ± 0.0201 | 0.6804 ± 0.0357 | -0.1961 | 0.6909 ± 0.0319 | 0.3352 ± 0.0278 |
| **ResNet-18** | `finetune_robust` | **1/5 seeds** | **⏳ Na fila (GPU 0)** | **0.8366** | **0.7587** | **-0.0779** | **0.6451** | **0.3651** |
| MobileNetV3-Large | `finetune` (Baseline) | 6 seeds | ✅ Concluído | 0.8454 ± 0.0041 | 0.6832 ± 0.0299 | -0.1622 | 0.7037 ± 0.0140 | 0.3804 ± 0.0115 |
| **MobileNetV3-Large** | `finetune_robust` | **1/5 seeds** | **⏳ Na fila (GPU 1)** | **0.8373** | **0.7493** | **-0.0880** | **0.7304** | **0.3728** |
| Xception | `finetune` (Baseline) | 6 seeds | ✅ Concluído | 0.7708 ± 0.0058 | 0.6388 ± 0.0202 | -0.1321 | 0.7298 ± 0.0244 | 0.3979 ± 0.0144 |
| **Xception** | `finetune_robust` | **1/5 seeds** | **⏳ Na fila (GPU 1)** | **0.7605** | **0.6838** | **-0.0767** | **0.6814** | **0.4177** |

---

## 2. Detalhamento por Semente Individual (`finetune_robust`)

Apresenta as métricas exatas obtidas em cada semente avaliada até o momento:

| Modelo | Seed | Status | Test AUC | Test ACC | Test F1 | Test-D AUC | Test-D ACC | Test-D F1 | ΔAUC | DF-40 AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | `seed_123` | ✅ Concluído | 0.9101 | 0.8208 | 0.8289 | **0.8492** | 0.7511 | 0.7496 | -0.0609 | 0.8049 |
| **CLIP ViT-B/16** | `seed_2024` | ✅ Concluído | 0.9075 | 0.8138 | 0.8178 | **0.8434** | 0.7416 | 0.7371 | -0.0641 | 0.8325 |
| **CLIP ViT-B/16** | `seed_42` | ✅ Concluído | 0.9066 | 0.8156 | 0.8220 | **0.8459** | 0.7465 | 0.7432 | -0.0607 | 0.8160 |
| **CLIP ViT-B/16** | `seed_987` | ✅ Concluído | 0.9067 | 0.8174 | 0.8231 | **0.8462** | 0.7398 | 0.7298 | -0.0604 | 0.8263 |
| *CLIP ViT-B/16* | `seed_7` | 🔄 Em andamento (GPU 0) | - | - | - | - | - | - | - | - |
| **DINO (ConvNeXt-B)** | `seed_987` | ✅ Concluído | 0.9242 | 0.8454 | 0.8541 | **0.8483** | 0.7656 | 0.7720 | -0.0759 | 0.7517 |
| *DINO (ConvNeXt-B)* | `seed_42` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| *DINO (ConvNeXt-B)* | `seed_123` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| *DINO (ConvNeXt-B)* | `seed_2024` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| *DINO (ConvNeXt-B)* | `seed_7` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| **Vision Transformer (ViT-B/16)** | `seed_123` | ✅ Concluído | 0.8253 | 0.7383 | 0.7465 | **0.7647** | 0.6850 | 0.6938 | -0.0605 | 0.7118 |
| **Vision Transformer (ViT-B/16)** | `seed_42` | ✅ Concluído | 0.8200 | 0.7363 | 0.7468 | **0.7619** | 0.6843 | 0.6939 | -0.0581 | 0.7257 |
| **Vision Transformer (ViT-B/16)** | `seed_987` | ✅ Concluído | 0.8176 | 0.7331 | 0.7421 | **0.7596** | 0.6812 | 0.6908 | -0.0580 | 0.7272 |
| *Vision Transformer (ViT-B/16)* | `seed_2024` | 🔄 Em andamento (GPU 1) | - | - | - | - | - | - | - | - |
| *Vision Transformer (ViT-B/16)* | `seed_7` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| **ResNet-18** | `seed_987` | ✅ Concluído | 0.8366 | 0.7369 | 0.7379 | **0.7587** | 0.6622 | 0.6535 | -0.0779 | 0.6451 |
| *ResNet-18* | `seed_42` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| *ResNet-18* | `seed_123` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| *ResNet-18* | `seed_2024` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| *ResNet-18* | `seed_7` | ⏳ Na fila (GPU 0) | - | - | - | - | - | - | - | - |
| **MobileNetV3-Large** | `seed_987` | ✅ Concluído | 0.8373 | 0.7370 | 0.7345 | **0.7493** | 0.6646 | 0.6635 | -0.0880 | 0.7304 |
| *MobileNetV3-Large* | `seed_42` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| *MobileNetV3-Large* | `seed_123` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| *MobileNetV3-Large* | `seed_2024` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| *MobileNetV3-Large* | `seed_7` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| **Xception** | `seed_987` | ✅ Concluído | 0.7605 | 0.6965 | 0.7419 | **0.6838** | 0.6514 | 0.7095 | -0.0767 | 0.6814 |
| *Xception* | `seed_42` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| *Xception* | `seed_123` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| *Xception* | `seed_2024` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |
| *Xception* | `seed_7` | ⏳ Na fila (GPU 1) | - | - | - | - | - | - | - | - |