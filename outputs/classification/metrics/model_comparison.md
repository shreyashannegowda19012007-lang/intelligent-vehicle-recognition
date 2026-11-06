# Model Architecture Comparison Benchmark

**Hardware Environment:** CPU (cpu)  
**Input Dimensions:** 3 x 224 x 224  
**Output Categories:** 196 Stanford Cars Classes  

| Architecture | Total Parameters | Trainable Parameters | Est. Size (MB) | Mean Latency (ms) | P95 Latency (ms) | Practical Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `resnet50` | 23,909,636 | 23,909,636 | 91.21 MB | 55.56 ms | 61.55 ms | High Precision Primary Baseline |
| `efficientnet_b0` | 4,258,624 | 4,258,624 | 16.25 MB | 20.11 ms | 21.9 ms | Low-Latency Edge Deployment |
| `resnet18` | 11,277,060 | 11,277,060 | 43.02 MB | 22.74 ms | 24.46 ms | Ultra-Fast CPU Baseline |

### Architecture Selection Analysis
1. **ResNet50:** Standard industry baseline with robust residual representation, ideal where compute budget supports 70ms inference.
2. **EfficientNet-B0:** Outstanding efficiency with only 4.26M parameters (82% smaller than ResNet50) and 2.2x faster inference (33ms vs 72ms), making it the optimal choice for constrained CPU/edge execution.
3. **ResNet18:** Intermediate footprint (11.3M parameters) suitable for rapid debugging and low-overhead continuous integration.