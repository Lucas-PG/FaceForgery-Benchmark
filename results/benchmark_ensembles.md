# Benchmark de Ensembles, Fusões Multimodais e Super-Ensembles

Este documento consolida o desempenho das estratégias de combinação por agregação de probabilidades (Soft Voting e Logistic Regression):

---

## 1. Tabela Geral de Ensembles

| Tipo de Ensemble | Modelos Componentes | Domínio | Test AUC (FF++) | Test-D AUC (Corrompido) | DF-40 AUC (Cross-Gen) | Celeb-DF Frame AUC | Celeb-DF Vídeo AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Super-Ensemble Multimodal** | CLIP (RGB) + DINO (RGB) + ResNet (Concat Freq) | `Espaço + Espectro` | 0.9512 | **0.8580** | **0.8654** | 0.7124 | 0.7512 |
| **Super-Ensemble 4-Peritos** | CLIP (RGB) + DINO (RGB) + ResNet (Freq) + ViT (Freq) | `Espaço + Espectro` | 0.9540 | **0.8610** | **0.8687** | 0.7180 | 0.7590 |
| **Ensemble Robusto Duplo** | CLIP Robusto + DINO Robusto | `RGB Robusto` | 0.9420 | **0.8712** | **0.8492** | 0.7450 | 0.7812 |
| **Ensemble Robusto 6 Modelos** | CLIP + DINO + ViT + ResNet + MobileNet + Xception | `RGB Robusto` | 0.9485 | **0.8845** | **0.8610** | 0.7590 | 0.7940 |
| **Ensemble Espacial 6x** | Todos os 6 modelos em RGB Puro | `Espaço RGB` | 0.9460 | **0.7745** | **0.8519** | 0.6845 | 0.7012 |
| **Ensemble Espectral 6x** | Todos os 6 modelos em Concat Frequency | `Espectro 7C` | 0.8920 | **0.7180** | **0.7915** | 0.5510 | 0.5620 |
| **MoE Standard** | 7 Peritos Convolucionais com Roteador Dinâmico | `Espaço RGB` | 0.8773 | **0.6796** | **0.7420** | 0.6120 | 0.6340 |
| **Frequency MoE** | 7 Peritos FFT com Roteador Espectral | `Espectro 7C` | 0.8737 | **0.6650** | **0.7250** | 0.5980 | 0.6150 |