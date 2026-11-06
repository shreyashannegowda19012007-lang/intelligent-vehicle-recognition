# Fine-Grained Vehicle Classification — Evaluation Report

**Architecture:** resnet50  
**Checkpoint:** `outputs\classification\checkpoints\best_checkpoint.pt`  
**Device:** cuda  
**Evaluated Samples:** 1222  
**Average Latency:** 19.41 ms/image  

## 1. Classification Metrics (196 Stanford Cars Classes)

- **Top-1 Accuracy:** `81.01%`
- **Top-5 Accuracy:** `95.17%`
- **Macro Precision:** `0.8376`
- **Weighted Precision:** `0.8369`
- **Macro Recall:** `0.8077`
- **Weighted Recall:** `0.8101`
- **Macro F1-Score:** `0.8024`
- **Weighted F1-Score:** `0.8040`

## 2. Decoded Attribute-Level Accuracies

> Note: Make, Model, and Year are decoded from the primary 196-class prediction using the class mapping.

- **Make Accuracy:** `89.61%` (49 makes)
- **Model Accuracy:** `83.31%`
- **Year Accuracy:** `90.10%`

## 3. Confidence Statistics

- **Mean Confidence:** `0.6548`
- **Median Confidence:** `0.7605`
- **Min / Max:** `0.0393` / `0.9929`
- **25th / 75th Percentile:** `0.4028` / `0.9139`

## 4. Top Confused Vehicle Classes

| True Vehicle Class | Predicted Vehicle Class | Count |
| :--- | :--- | :--- |
| Audi A5 Coupe 2012 | Audi S5 Coupe 2012 | 3 |
| Audi 100 Sedan 1994 | Audi 100 Wagon 1994 | 3 |
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 3 |
| Ferrari 458 Italia Coupe 2012 | Ferrari 458 Italia Convertible 2012 | 3 |
| Acura RL Sedan 2012 | Acura TL Sedan 2012 | 2 |
| Aston Martin V8 Vantage Convertible 2012 | Aston Martin V8 Vantage Coupe 2012 | 2 |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 2 |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 2 |
| Audi TT Hatchback 2011 | Audi TT RS Coupe 2012 | 2 |
| Audi S4 Sedan 2007 | Audi RS 4 Convertible 2008 | 2 |

## 5. Diagnostic Artifacts & Visualizations

- **Full Confusion Matrix:** `outputs/classification/metrics/confusion_matrix.csv`
- **Top Confused Plot:** `outputs/classification/plots/confusion_matrix_top20.png`
- **Confidence Histogram:** `outputs/classification/plots/confidence_distribution.png`
- **Training Curves:** `outputs/classification/plots/training_history.png`
- **Sample Predictions:** `outputs/classification/predictions/sample_predictions.json`
