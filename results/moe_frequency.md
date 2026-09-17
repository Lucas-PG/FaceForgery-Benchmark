# Resultados do Modelo: Mixture of Experts Frequencial (`moe_frequency`)

> **Arquitetura**: Mixture of Experts (MoE) com **7 especialistas dedicados** baseados em MobileNet-Small e roteador Top-3 denso.
> Especialistas operam em domínios complementares: Spatial RGB, Magnitude FFT, Phase FFT, High-Pass Filter, Low-Pass Filter, Residual High-Pass e Edge Magnitude.

## 1. Especificações Arquiteturais

| Parâmetro | Configuração | Descrição |
| :--- | :--- | :--- |
| **Família do Modelo** | `moe_frequency` | Mixture of Experts no domínio de frequência |
| **Backbone dos Especialistas** | `mobilenet` (Small) | 7 redes neurais convolucionais independentes |
| **Número de Especialistas** | `7` | 1 Espacial RGB + 6 Representações Fourier 2D-FFT |
| **Estratégia de Roteamento** | `dense` (`top_k=3`) | Gating pondera os 3 especialistas com maior ativação |
| **Entradas dos Especialistas** | 7 canais/visões | RGB, Magnitude, Fase, Passa-Alta, Passa-Baixa, Residual, Bordas |
| **Regime de Treinamento** | `scratch` (v1) e `scratch_robust` (v2) | Treinado de ponta a ponta por 20 épocas |
| **Função de Perda / Otimizador** | Cross-Entropy + AdamW | LR Head: 1e-3, LR Backbone: 1e-4, Weight Decay: 1e-4 |

## 2. Desempenho no Benchmark Fase 1 (Semente 42)

| Split de Avaliação | AUC | Acurácia (ACC) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) | Limiar Ótimo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validação (`val`)** | **0.9705** | 0.9144 | 0.9270 | 0.9477 | 0.9072 | 0.9251 | 0.5639 | `0.9885` |
| **Teste Padrão (`test`)** | **0.9068** | 0.8253 | 0.8362 | 0.9040 | 0.7779 | 0.8889 | 1.5049 | `0.9885` |
| **Teste Severo (`test_d`)** | **0.6512** | 0.6240 | 0.6812 | 0.6629 | 0.7005 | 0.5211 | 5.6885 | `0.9885` |

## 3. Análise de Robustez e Queda de Desempenho (Test vs Test_d)

| Cenário | Test Padrão | Test Perturbado (`test_d`) | Degradação Absoluta (Δ) | Retenção (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.9068 | 0.6512 | `-0.2556` | `71.81%` |
| **Acurácia (ACC)** | 0.8253 | 0.6240 | `-0.2013` | `75.61%` |
| **F1-Score** | 0.8362 | 0.6812 | `-0.1550` | `81.46%` |

## 4. Evolução Arquitetural: MoE 4 Especialistas vs MoE 7 Especialistas

| Split | Métrica | MoE 4 Especialistas (Linha de Base) | MoE 7 Especialistas (Atual) | Ganho Absoluto (Δ) | Ganho Relativo (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `val` | **AUC** | `0.9496` | `0.9705` | `+0.0209` | `+2.20%` |
| `val` | **Acurácia** | `0.8800` | `0.9144` | `+0.0344` | `+3.91%` |
| `val` | **F1-Score** | `0.8964` | `0.9270` | `+0.0306` | `+3.41%` |
| `val` | **Loss** | `0.8365` | `0.5639` | `-0.2726` | `-32.59%` |
| `test` | **AUC** | `0.8737` | `0.9068` | `+0.0331` | `+3.79%` |
| `test` | **Acurácia** | `0.7868` | `0.8253` | `+0.0385` | `+4.89%` |
| `test` | **F1-Score** | `0.7969` | `0.8362` | `+0.0393` | `+4.93%` |
| `test` | **Loss** | `1.7828` | `1.5049` | `-0.2779` | `-15.59%` |

## 5. Mapeamento e Papel dos 7 Especialistas Frequenciais

| ID Especialista | Domínio de Entrada | Resolução | Propósito Forense |
| :---: | :--- | :---: | :--- |
| **Expert 0** | Spatial RGB | 224x224x3 | Análise contextual, artefatos semânticos e consistência anatômica |
| **Expert 1** | Magnitude FFT 2D | 224x224x1 | Distribuição de energia espectral, picos periódicos e grade de interpolação |
| **Expert 2** | Fase FFT 2D | 224x224x1 | Relações espaciais de bordas e coerência de fase |
| **Expert 3** | Passa-Alta (High-Pass) | 224x224x1 | Resíduos de alta frequência, descontinuidades finas e ruído de compressão |
| **Expert 4** | Passa-Baixa (Low-Pass) | 224x224x1 | Estrutura macro, iluminação global e transições suaves de gradiente |
| **Expert 5** | Residual Espectral | 224x224x1 | Discrepância entre espectro original e aproximação suavizada |
| **Expert 6** | Magnitude de Bordas (Sobel) | 224x224x1 | Fronteiras de splicing, máscaras de blending e artefatos de boundary |
