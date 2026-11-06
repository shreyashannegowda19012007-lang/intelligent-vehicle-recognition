# Fine-Grained Vehicle Classification — Evaluation Report

**Architecture:** resnet50  
**Checkpoint:** `outputs\classification\checkpoints\best_checkpoint.pt`  
**Device:** cpu  
**Evaluated Samples:** 1222  
**Average Latency:** 72.14 ms/image  

## 1. Classification Metrics (196 Stanford Cars Classes)

- **Top-1 Accuracy:** `0.49%`
- **Top-5 Accuracy:** `2.37%`
- **Macro Precision:** `0.0000`
- **Weighted Precision:** `0.0000`
- **Macro Recall:** `0.0051`
- **Weighted Recall:** `0.0049`
- **Macro F1-Score:** `0.0000`
- **Weighted F1-Score:** `0.0000`

## 2. Decoded Attribute-Level Accuracies

> Note: Make, Model, and Year are decoded from the primary 196-class prediction using the class mapping.

- **Make Accuracy:** `7.12%` (49 makes)
- **Model Accuracy:** `1.06%`
- **Year Accuracy:** `58.92%`

## 3. Confidence Statistics

- **Mean Confidence:** `0.0070`
- **Median Confidence:** `0.0070`
- **Min / Max:** `0.0070` / `0.0070`
- **25th / 75th Percentile:** `0.0070` / `0.0070`

## 4. Top Confused Vehicle Classes

| True Vehicle Class | Predicted Vehicle Class | Count |
| :--- | :--- | :--- |
| GMC Savana Van 2012 | Audi S4 Sedan 2012 | 10 |
| AM General Hummer SUV 2000 | Audi S4 Sedan 2012 | 7 |
| Acura Integra Type R 2001 | Audi S4 Sedan 2012 | 7 |
| Aston Martin V8 Vantage Convertible 2012 | Audi S4 Sedan 2012 | 7 |
| Audi V8 Sedan 1994 | Audi S4 Sedan 2012 | 7 |
| Audi S6 Sedan 2011 | Audi S4 Sedan 2012 | 7 |
| Audi S4 Sedan 2007 | Audi S4 Sedan 2012 | 7 |
| BMW 6 Series Convertible 2007 | Audi S4 Sedan 2012 | 7 |
| BMW M3 Coupe 2012 | Audi S4 Sedan 2012 | 7 |
| Bentley Continental GT Coupe 2007 | Audi S4 Sedan 2012 | 7 |

## 5. Diagnostic Artifacts & Visualizations

- **Full Confusion Matrix:** `outputs/classification/metrics/confusion_matrix.csv`
- **Top Confused Plot:** `outputs/classification/plots/confusion_matrix_top20.png`
- **Confidence Histogram:** `outputs/classification/plots/confidence_distribution.png`
- **Training Curves:** `outputs/classification/plots/training_history.png`
- **Sample Predictions:** `outputs/classification/predictions/sample_predictions.json`
