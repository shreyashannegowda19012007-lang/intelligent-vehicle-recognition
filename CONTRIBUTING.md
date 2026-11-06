# Contributing to Intelligent Fine-Grained Vehicle Recognition System

Thank you for your interest in contributing to the **Intelligent Fine-Grained Vehicle Recognition and Analysis System**. This document outlines our collaborative development workflow, coding standards, testing requirements, and asset management rules.

---

## 1. Collaboration Workflow

All development follows a strict branch-and-PR model:

```
main ──> feature branch ──> Pull Request ──> Code Review & CI ──> Merge into main
```

Direct pushes to the `main` branch are restricted. All changes must be proposed through pull requests, pass automated test verification, and receive review approval before merging.

### Standard Branch Naming Conventions

Use one of the following branch naming conventions depending on your task:

| Branch Name Prefix | Purpose | Example |
| :--- | :--- | :--- |
| `feature/classical-cv` | Classical computer vision feature engineering | `feature/classical-cv-orb-tuning` |
| `feature/yolo-detection` | YOLO vehicle detection and ROI extraction | `feature/yolo-detection-nms-tuning` |
| `feature/deep-learning` | Deep classification architectures (ConvNeXt, ResNet) | `feature/deep-learning-convnext-backbone` |
| `feature/hybrid-system` | Classical + deep feature fusion heads | `feature/hybrid-system-cross-attention` |
| `feature/frontend` | Interactive web dashboard and UI | `feature/frontend-streamlit-demo` |
| `feature/evaluation` | Evaluation metrics, ablation studies, reports | `feature/evaluation-confusion-matrix` |
| `docs/documentation` | Documentation updates and guides | `docs/documentation-api-reference` |
| `fix/` | Bug fixes for existing functionality | `fix/bbox-coordinate-rounding` |

---

## 2. Step-by-Step Contribution Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/shreyashannegowda19012007-lang/intelligent-vehicle-recognition.git
cd intelligent-vehicle-recognition
```

### Step 2: Set Up the Development Environment

Create and activate a Python 3.11 virtual environment:

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
# For CUDA 12.1 acceleration:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# For CPU-only development:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Copy the example environment configuration:
```bash
cp .env.example .env
```

### Step 3: Create a Feature Branch

Ensure your local `main` branch is up to date, then branch off:

```bash
git checkout main
git pull origin main
git checkout -b feature/deep-learning-convnext
```

### Step 4: Implement Changes and Write Tests

- Write clean, modular Python following PEP 8 conventions.
- Add type annotations (`typing` or Python 3.10+ native types) for all public functions and classes.
- Write docstrings for all modules, classes, and methods.
- Accompany every new feature or bug fix with unit tests in `tests/`.

### Step 5: Run Tests Locally

Before committing, run the test suite to ensure no regressions:

```bash
# Run lightweight unit tests (no dataset required, 48 tests)
pytest tests/test_initialization.py tests/test_annotations.py tests/test_classical_cv.py tests/test_preprocessing.py tests/test_splits.py -v

# Run the complete test suite (when Stanford Cars dataset is available locally)
pytest tests/ -v
```

All existing tests must pass before opening a pull request.

### Step 6: Commit Your Changes

Follow conventional commit messages:

```bash
git add src/classification/backbone.py tests/test_classification.py
git commit -m "feat(classification): Implement ConvNeXt-Tiny fine-grained feature extractor"
```

Common prefixes:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `test`: Adding or correcting tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement

### Step 7: Push and Open a Pull Request

Push your feature branch to GitHub:

```bash
git push -u origin feature/deep-learning-convnext
```

Then open a Pull Request against `main` on GitHub:
1. Provide a descriptive PR title and summary of changes.
2. Link any related issues or milestones.
3. Verify that GitHub Actions CI checks pass.
4. Request review from team members.

---

## 3. Strict Rules: Datasets & Model Weights

To maintain repository health, prevent Git bloat, and comply with academic licenses:

1. **NO Dataset Images**: NEVER commit raw or processed image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) from the Stanford Cars dataset or any other image dataset. All image directories are excluded via `.gitignore`.
2. **NO Large Binary Weights**: NEVER commit model weight binaries (`.pt`, `.pth`, `.joblib`, `.onnx`, `.engine`, `.bin`). Model checkpoints must be saved locally or published to an external artifact store (e.g., GitHub Releases, Hugging Face Hub, or cloud buckets).
3. **NO Secrets or Credentials**: NEVER commit `.env`, API keys, tokens, or credentials.
4. **NO Virtual Environments or Caches**: Keep `.venv/`, `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/` strictly excluded.

---

## 4. Collaborator Access and Permissions

To collaborate on this private repository:
- Repository owner manages access via **GitHub Repository Settings → Collaborators**.
- Contact the repository administrator with your GitHub username to receive an invitation.
- Branch protection rules on `main` require PR reviews and passing CI checks prior to merging.

