# Resultados do Modelo: Mixture of Experts Padrão (`moe_standard`)

> **Arquitetura**: Mixture of Experts (MoE) com **7 especialistas espaciais** baseados em MobileNet-Small e roteador Top-3 denso.
> Especialistas operam diretamente no domínio espacial RGB puro (`fourier_mode: none`).

## 1. Especificações Arquiteturais

| Parâmetro | Configuração | Descrição |
| :--- | :--- | :--- |
| **Família do Modelo** | `moe_standard` | Mixture of Experts no domínio puramente espacial RGB |
| **Backbone dos Especialistas** | `mobilenet` (Small) | 7 redes neurais convolucionais independentes |
| **Número de Especialistas** | `7` | 7 especialistas espaciais RGB |
| **Estratégia de Roteamento** | `dense` (`top_k=3`) | Gating pondera os 3 especialistas com maior ativação |
| **Entradas dos Especialistas** | 3 canais (RGB padrão) | Imagem redimensionada 224x224 normalizada ImageNet |
| **Regime de Treinamento** | `scratch` (v1) e `scratch_robust` (v2) | Treinado de ponta a ponta por 20 épocas |
| **Função de Perda / Otimizador** | Cross-Entropy + AdamW | LR Head: 1e-3, LR Backbone: 1e-4, Weight Decay: 1e-4 |

## 2. Desempenho no Benchmark Fase 1 (Semente 42)

| Split de Avaliação | AUC | Acurácia (ACC) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) | Limiar Ótimo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validação (`val`)** | **0.9316** | 0.8478 | 0.8656 | 0.9186 | 0.8184 | 0.8917 | 0.6767 | `0.9774` |
| **Teste Padrão (`test`)** | **0.8773** | 0.7773 | 0.7777 | 0.9098 | 0.6790 | 0.9095 | 1.1193 | `0.9774` |
| **Teste Severo (`test_d`)** | **0.6796** | 0.6199 | 0.6235 | 0.7218 | 0.5488 | 0.7156 | 2.3879 | `0.9774` |

## 3. Análise de Robustez e Queda de Desempenho (Test vs Test_d)

| Cenário | Test Padrão | Test Perturbado (`test_d`) | Degradação Absoluta (Δ) | Retenção (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.8773 | 0.6796 | `-0.1976` | `77.47%` |
| **Acurácia (ACC)** | 0.7773 | 0.6199 | `-0.1574` | `79.75%` |
| **F1-Score** | 0.7777 | 0.6235 | `-0.1541` | `80.18%` |

## 4. Comparação: MoE 4 Especialistas vs MoE 7 Especialistas

| Split | Métrica | MoE 4 Especialistas (Linha de Base) | MoE 7 Especialistas (Atual) | Ganho Absoluto (Δ) | Ganho Relativo (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `val` | **AUC** | `0.9332` | `0.9316` | `-0.0016` | `-0.17%` |
| `val` | **Acurácia** | `0.8563` | `0.8478` | `-0.0085` | `-0.99%` |
| `val` | **F1-Score** | `0.8725` | `0.8656` | `-0.0069` | `-0.79%` |
| `val` | **Loss** | `0.7293` | `0.6767` | `-0.0526` | `-7.21%` |
| `test` | **AUC** | `0.8831` | `0.8773` | `-0.0058` | `-0.66%` |
| `test` | **Acurácia** | `0.7788` | `0.7773` | `-0.0015` | `-0.19%` |
| `test` | **F1-Score** | `0.7775` | `0.7777` | `+0.0002` | `+0.03%` |
| `test` | **Loss** | `1.1791` | `1.1193` | `-0.0598` | `-5.07%` |

## 5. Comparativo Direto: MoE Standard (Espacial) vs MoE Frequency (Fourier)

| Split de Avaliação | Métrica | MoE Standard (RGB) | MoE Frequency (Fourier) | Vencedor | Vantagem Frequencial |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `val` | **AUC** | `0.9316` | `0.9705` | **MoE Frequency** | `+0.0389 (+4.18%)` |
| `val` | **Acurácia** | `0.8478` | `0.9144` | **MoE Frequency** | `+0.0666 (+7.86%)` |
| `val` | **Loss** | `0.6767` | `0.5639` | **MoE Frequency** | `-0.1128 (-16.67%)` |
| `test` | **AUC** | `0.8773` | `0.9068` | **MoE Frequency** | `+0.0295 (+3.36%)` |
| `test` | **Acurácia** | `0.7773` | `0.8253` | **MoE Frequency** | `+0.0480 (+6.17%)` |
| `test_d` | **AUC** | `0.6796` | `0.6512` | **MoE Standard** | `-0.0284 (-4.18%)` |
| `test_d` | **Loss** | `2.3879` | `5.6885` | **MoE Standard** | `+3.3006 (+138.2%)` |
