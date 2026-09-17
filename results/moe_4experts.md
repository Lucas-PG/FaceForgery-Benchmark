# Resultados dos Modelos MoE Históricos: Linha de Base com 4 Especialistas

> **Histórico Arquitetural**: Primeira geração dos modelos Mixture of Experts avaliados no projeto, configurados com **4 especialistas** (MobileNet-Small) e roteador Top-2 denso.
> Armazenados em `models/archive_4experts/` como ponto de referência para a expansão para 7 especialistas.

## 1. Tabela Comparativa de Desempenho dos Modelos com 4 Especialistas

| Modelo MoE | Modo | Split | AUC | Acurácia (ACC) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `moe_frequency` (4 exp) | `concat_frequency` | `val` | **0.9496** | 0.8800 | 0.8964 | 0.9282 | 0.8668 | 0.8998 | 0.8365 |
| `moe_frequency` (4 exp) | `concat_frequency` | `test` | **0.8737** | 0.7868 | 0.7969 | 0.8784 | 0.7292 | 0.8642 | 1.7828 |
| `moe_standard` (4 exp) | `none` | `val` | **0.9332** | 0.8563 | 0.8725 | 0.9313 | 0.8208 | 0.9095 | 0.7293 |
| `moe_standard` (4 exp) | `none` | `test` | **0.8831** | 0.7788 | 0.7775 | 0.9187 | 0.6739 | 0.9198 | 1.1791 |

## 2. Comparação Direta entre MoE Frequencial e Espacial (4 Especialistas)

| Split | Métrica | MoE Frequency (4 exp) | MoE Standard (4 exp) | Vantagem Frequencial (Δ) |
| :--- | :--- | :---: | :---: | :---: |
| `val` | **AUC** | `0.9496` | `0.9332` | `+0.0164 (+1.76%)` |
| `val` | **Acurácia** | `0.8800` | `0.8563` | `+0.0237 (+2.77%)` |
| `val` | **F1-Score** | `0.8964` | `0.8725` | `+0.0239 (+2.74%)` |
| `test` | **AUC** | `0.8737` | `0.8831` | `-0.0094 (-1.06%)` |
| `test` | **Acurácia** | `0.7868` | `0.7788` | `+0.0080 (+1.03%)` |
| `test` | **F1-Score** | `0.7969` | `0.7775` | `+0.0194 (+2.50%)` |

## 3. Configurações de Hiperparâmetros (MoE 4 Especialistas)

| Parâmetro | `moe_frequency` (4 exp) | `moe_standard` (4 exp) |
| :--- | :---: | :---: |
| **Número de Especialistas** | 4 | 4 |
| **Top-K Roteamento** | 2 | 2 |
| **Canais de Entrada** | 6 (RGB + Magnitude + Fase + HPF) | 3 (RGB) |
| **Backbone** | MobileNetV3-Small | MobileNetV3-Small |
| **Épocas** | 20 | 20 |
| **Batch Size** | 16 | 16 |
| **Limiar Ótimo de Decisão** | `0.9885` | `0.9774` |
