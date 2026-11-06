# Google Colab Training & Integration Guide: Stanford Cars ResNet50

This guide provides step-by-step instructions to train, evaluate, and integrate a high-accuracy torchvision **ResNet50** model for fine-grained vehicle recognition (Stanford Cars 196 classes) using Google Colab or any cloud GPU instance.

---

## 1. Prerequisites & Cloud GPU Selection

### 1.1 Open Notebook in Google Colab
1. Upload [`notebooks/train_stanford_cars_resnet50.ipynb`](file:///e:/skanda/RVU/intelligent-vehicle-recognition/notebooks/train_stanford_cars_resnet50.ipynb) to Google Drive or open directly in [Google Colab](https://colab.research.google.com).
2. Configure GPU Runtime:
   - In Google Colab, navigate to **Runtime** > **Change runtime type**.
   - Under **Hardware accelerator**, select **GPU**.
   - Under **GPU type**, choose in order of preference:
     - **NVIDIA L4** (Best price/performance, 24 GB VRAM, Ada Lovelace, native BF16 support)
     - **NVIDIA T4** (Standard free/Pro tier, 15 GB VRAM, Turing, native FP16 support)
     - **NVIDIA A100** (High-memory tier, 40/80 GB VRAM, Ampere, native BF16 support)

The notebook automatically inspects `torch.cuda.get_device_properties(0)` and tunes:
- Batch Size: 64 for L4/A100; 32 for T4
- Mixed Precision: `torch.bfloat16` for L4/A100; `torch.float16` for T4
- DataLoader workers and `torch.channels_last` memory layout

---

## 2. Execution Steps in Colab

### Step 1: Run Environment Check & Mount Google Drive
Execute **Cell 1** and **Cell 2**:
- Connects Google Drive at `/content/drive`.
- Establishes persistent checkpoint directory at `/content/drive/MyDrive/stanford_cars_checkpoints/`.
- If an interrupted training run was saved previously, it detects `last_checkpoint.pt` and automatically resumes from the exact epoch.

### Step 2: Kaggle Dataset Ingestion
Execute **Cell 3**:
- Stanford Cars (196 classes) dataset is downloaded via Kaggle API (`eduardo4jesus/stanford-cars-dataset`).
- **Authentication**:
  - Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab **Secrets** (key icon on the left panel), OR
  - Upload your `kaggle.json` when prompted.
- The dataset is extracted to `/content/dataset/` with all 196 class folders in `train` and `test`.

### Step 3: Taxonomy Alignment & DataLoaders
Execute **Cell 4**:
- Maps all 196 classes to canonical IDs `0..195`.
- Normalizes filesystem edge cases (e.g. `Ram C_V Cargo Van Minivan 2012` <-> `Ram C/V Cargo Van Minivan 2012`).
- Initializes GPU DataLoaders with `pin_memory=True` and ImageNet standard normalization:
  - Training: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip()`, `ColorJitter()`
  - Validation: Deterministic `Resize(256)` + `CenterCrop(224)`

### Step 4: ResNet50 2-Stage Training
Execute **Cell 5** and **Cell 6**:
- **Stage A (Warmup - 2 Epochs)**:
  - Backbone parameters are frozen (`requires_grad=False`).
  - Only the 196-class Linear head trains with learning rate `5e-4`.
- **Stage B (Fine-Tuning - up to 28 Epochs)**:
  - All layers unfrozen (`requires_grad=True`).
  - Discriminative learning rates: `1e-4` for backbone, `5e-4` for head.
  - Optimizer: `AdamW` with weight decay `1e-4`, Cosine Annealing scheduler, and label smoothing `0.05`.
  - Automatic Early Stopping: stops if validation Top-1 accuracy plateaus for 5 consecutive epochs.
- **Persistence**:
  - After every epoch, `last_checkpoint.pt` and `best_checkpoint.pt` are synced to Google Drive.

### Step 5: Full Multi-Attribute Evaluation
Execute **Cell 7**:
- Loads `best_checkpoint.pt` and evaluates across the validation set.
- Computes:
  - **Top-1 & Top-5 Classification Accuracy**
  - **Make, Model & Year Attribute Accuracy**
  - **Inference Latency (ms per image)**
  - **Model Checkpoint Size (MB)**
- Generates and displays training/validation loss and accuracy curves (`training_history.png`).

### Step 6: Download Artifact Package
Execute **Cell 8**:
- Bundles all outputs into `stanford_cars_resnet50_artifacts.zip`:
  - `checkpoints/best_checkpoint.pt`
  - `metrics/class_mapping.json`
  - `metrics/eval_results.json`
  - `metrics/training_history.json`
  - `plots/training_history.png`
- Automatically triggers browser download and backs up the ZIP to Google Drive.

---

## 3. Integrating Trained Checkpoint into Local Project

Once `stanford_cars_resnet50_artifacts.zip` is downloaded:

1. Unzip the contents into your project's `outputs/classification/` folder:
   ```text
   outputs/classification/
   ├── checkpoints/
   │   └── best_checkpoint.pt
   ├── metrics/
   │   ├── class_mapping.json
   │   ├── eval_results.json
   │   └── training_history.json
   └── plots/
       └── training_history.png
   ```

2. Verify Checkpoint Detection:
   ```powershell
   python scripts/download_models.py --check-only
   ```
   Output confirms:
   ```text
   Model [resnet50_stanford_cars]: FOUND (outputs/classification/checkpoints/best_checkpoint.pt)
   ```

3. Run End-to-End Application Verification:
   ```powershell
   python scripts/verify_e2e.py
   ```
   The `DeepLearningAdapter` and `HybridVehiclePipeline` automatically detect the trained checkpoint and switch from baseline fallback to live fine-tuned classification inference.

4. Start FastAPI & Streamlit Services:
   ```powershell
   uvicorn app.api:app --reload --port 8000
   streamlit run app/streamlit_app.py
   ```

