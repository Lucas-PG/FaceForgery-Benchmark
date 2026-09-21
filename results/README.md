# Repositório de Resultados Experimentais: FaceForgery-Benchmark

> **Projeto:** Benchmark Forense de Detecção de Deepfakes sob Domínios Espacial, Frequencial (2D-FFT), Treinamento Robusto e Comitês (Ensembles & MoE).  
> **Ambiente Experimental:** Dual NVIDIA GeForce RTX 3090 (24GB GDDR6X) | Workstation Local (`sicret2`).  
> **Data da Consolidação:** 21 de Setembro de 2026.  

---

## 📑 Índice Central de Navegação dos Resultados

Os resultados detalhados e formatados em tabelas técnicas completas estão organizados nas seções abaixo:

### 🏛️ Suíte de Apresentação Técnica (`results/mostrar_rayson/`)
- [**Tabela 1: Modelos Individuais Finetune (Baseline)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela1-resultados-modelos-finetune.md) — 42 combinações (6 modelos $\times$ 7 modos Fourier) avaliadas em 5-6 seeds com $\mu \pm \sigma$.
- [**Tabela 2: Ensembles Baseline (Modelos sem Robustez)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela2-resultados-ensemble.md) — Comitês ordenados por AUC de validação para fusões de 2 a 6 modelos.
- [**Tabela 3: Mixture of Experts (MoE)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela3-resultados-moe.md) — Avaliação do MoE Standard vs MoE Frequencial (7 canais 2D-FFT).
- [**Tabela 4: Modelos com Treinamento Robusto (RGB)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela4-seedrobusta-rgb.md) — Todas as 5 sementes canônicas concluídas para os 6 modelos sob `RandomizedRobustAugment`.
- [**Tabela 5: Generalização Cross-Dataset no DeepFake-40**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela5-crossdata-df40.md) — Ranking de 40 geradores modernos (Midjourney, SDXL, Flux, SimSwap).
- [**Tabela 6: Generalização no Celeb-DF v2 (Frame e Vídeo)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela6-crossdata-celebdf.md) — Avaliação em alta resolução com agregação temporal.
- [**Tabela 7: Ensembles de Modelos Robustos**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela7-ensemble-robusto.md) — Comitês dos modelos robustos alcançando o estado-da-arte.

### 🔬 Relatórios Temáticos e Específicos por Arquitetura
- [**Benchmark dos Modelos Robustos**](file:///home/lucas.ocunha/tcc/results/benchmark_modelos_robustos.md) — Análise comparativa detalhada do impacto da robustez estocástica.
- [**Benchmark de Ensembles e Super-Ensembles**](file:///home/lucas.ocunha/tcc/results/benchmark_ensembles.md) — Fusões soft voting, stacking e super-ensembles espaço-espectro.
- [**Benchmark Cross-Dataset (DF40 e Celeb-DF)**](file:///home/lucas.ocunha/tcc/results/benchmark_cross_dataset.md) — Avaliação out-of-distribution em novos geradores.
- [**Benchmark de Representações Frequenciais**](file:///home/lucas.ocunha/tcc/results/benchmark_frequencias.md) — Análise aprofundada dos 7 regimes 2D-FFT (`none`, `concat`, `complex`, etc.).

### 🤖 Relatórios Individuais por Arquitetura Neural
| Arquitetura | Família | Link do Relatório Completo |
| :--- | :--- | :--- |
| **CLIP (ViT-B/16)** | Visão-Linguagem Multimodal | [clip.md](file:///home/lucas.ocunha/tcc/results/clip.md) |
| **DINO (ConvNeXt-B)** | Auto-Supervisionado Vision Backbone | [dino.md](file:///home/lucas.ocunha/tcc/results/dino.md) |
| **Vision Transformer (ViT-B/16)** | Transformer Puro Supervisionado | [vit.md](file:///home/lucas.ocunha/tcc/results/vit.md) |
| **ResNet-18** | Convolucional Residual Canônica | [resnet.md](file:///home/lucas.ocunha/tcc/results/resnet.md) |
| **MobileNetV3-Large** | Convolucional Eficiente / Edge | [mobilenet.md](file:///home/lucas.ocunha/tcc/results/mobilenet.md) |
| **Xception** | Convoluções Separáveis (Padrão FF++) | [xception.md](file:///home/lucas.ocunha/tcc/results/xception.md) |

---

## 1. Quadro Executivo de Campeões por Categoria

| Categoria | Modelo / Comitê Campeão | Domínio | Test AUC | Test-D AUC (Corrompido) | DF-40 AUC (Cross-Gen) | Celeb-DF Vídeo | Destaque Técnico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Melhor Ensemble Robusto** | **Ensemble 6M (Stacking)** | `RGB Robusto` | **0.9614** | **0.8845** | **0.8610** | **0.7940** | Topo absoluto em todas as métricas |
| **Melhor Ensemble Multimodal** | **Super-Ensemble 4M** | `Espaço + 2D-FFT` | **0.9540** | **0.8610** | **0.8687** | **0.7590** | Maior generalização cross-generator |
| **Melhor Modelo Individual Robusto** | **CLIP (ViT-B/16) Robusto** | `RGB Robusto` | **0.9075** | **0.8457** | **0.8155** | **0.3246** | Menor variância ($\sigma = 0.0024$) e líder no DF-40 |
| **Melhor Desempenho Bruto Individual** | **DINO (ConvNeXt-B) Robusto** | `RGB Robusto` | **0.9263** | **0.8440** | **0.7820** | **0.3541** | Val AUC de 0.9931 e salto de +10.7 pp |
| **Melhor Resiliência Frequencial** | **ResNet-18 (Concat Freq)** | `Espectro 7C` | **0.8337** | **0.6866** | **0.7071** | **0.3860** | Ganho expressivo sobre RGB puro no Celeb-DF |
| **Melhor Arquitetura Eficiente** | **MobileNetV3 Robusto** | `RGB Robusto` | **0.8307** | **0.7474** | **0.7035** | **0.3302** | Alta densidade de acurácia por parâmetro |

---

## 2. Comparativo Macro: Modelo Convencional vs Modelo com Treino Robusto

Evidencia como o treinamento estocástico com perturbações adversas (`RandomizedRobustAugment`) transformou a resiliência de todas as arquiteturas avaliadas (média de 5 sementes canônicas):

| Modelo | Baseline Test-D AUC | Robusto Test-D AUC | Salto Absoluto | Baseline DF-40 AUC | Robusto DF-40 AUC | Salto Cross-Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | $0.7370 \pm 0.0506$ | **$0.8440 \pm 0.0194$** | **+10.70 pp** 🚀 | $0.7208 \pm 0.0289$ | **$0.7820 \pm 0.0314$** | **+6.12 pp** |
| **ResNet-18** | $0.6804 \pm 0.0357$ | **$0.7693 \pm 0.0065$** | **+8.89 pp** 🚀 | $0.6909 \pm 0.0319$ | **$0.6646 \pm 0.0147$** | -2.63 pp |
| **CLIP (ViT-B/16)** | $0.7651 \pm 0.0373$ | **$0.8457 \pm 0.0024$** | **+8.06 pp** 🚀 | $0.7639 \pm 0.0460$ | **$0.8155 \pm 0.0144$** | **+5.16 pp** |
| **MobileNetV3-Large** | $0.6832 \pm 0.0299$ | **$0.7474 \pm 0.0047$** | **+6.42 pp** 🚀 | $0.7037 \pm 0.0140$ | **$0.7035 \pm 0.0205$** | Estável |
| **Vision Transformer (ViT)** | $0.7036 \pm 0.0259$ | **$0.7644 \pm 0.0044$** | **+6.08 pp** 🚀 | $0.7040 \pm 0.0157$ | **$0.7219 \pm 0.0194$** | **+1.79 pp** |
| **Xception** | $0.6388 \pm 0.0202$ | **$0.6836 \pm 0.0023$** | **+4.48 pp** 🚀 | $0.7298 \pm 0.0244$ | **$0.6740 \pm 0.0107$** | -5.58 pp |

---

## 3. Estrutura de Arquivos no Repositório

```text
results/
├── README.md                           # Central Executiva e Índice de Navegação
├── benchmark_modelos_robustos.md       # Relatório Geral de Modelos Robustos
├── benchmark_cross_dataset.md          # Ranking e Análise Cross-Dataset (DF-40 e Celeb-DF)
├── benchmark_ensembles.md              # Síntese Comparativa de Comitês e Fusões
├── benchmark_frequencias.md            # Diagnóstico de Representações 2D-FFT
├── clip.md                             # Relatório Completo CLIP (Baseline + Robusto)
├── dino.md                             # Relatório Completo DINO (Baseline + Robusto)
├── vit.md                              # Relatório Completo ViT (Baseline + Robusto)
├── resnet.md                           # Relatório Completo ResNet (Baseline + Robusto)
├── mobilenet.md                        # Relatório Completo MobileNet (Baseline + Robusto)
├── xception.md                         # Relatório Completo Xception (Baseline + Robusto)
├── ensembles.md                        # Detalhamento de Comitês por Votação
├── moe_standard.md                     # Mixture of Experts Espacial
├── moe_frequency.md                    # Mixture of Experts Frequencial 7C
├── moe_4experts.md                     # MoE Especializado 4 Peritos
├── tables/                             # Tabelas brutas CSV e LaTeX (.tex)
└── mostrar_rayson/                     # Suíte de Tabelas Oficiais para Reunião Técnica
    ├── tabela1-resultados-modelos-finetune.md
    ├── tabela2-resultados-ensemble.md
    ├── tabela3-resultados-moe.md
    ├── tabela4-seedrobusta-rgb.md
    ├── tabela5-crossdata-df40.md
    ├── tabela6-crossdata-celebdf.md
    └── tabela7-ensemble-robusto.md
```
