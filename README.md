# Cervical Spine Fracture AI Detection

This project implements a complete deep learning classification pipeline designed to screen and triage cervical spine fracture cases from 2D scans (such as axial/sagittal CT slices or radiographs). It includes dataset preprocessing, model training, evaluation, individual scan prediction, a REST API backend, and an interactive diagnostic web dashboard.

## Project Structure

```text
CervicalFractureAI/
│
├── dataset/
│   ├── train/
│   │   ├── fracture/
│   │   └── no_fracture/
│   │
│   ├── validation/
│   │   ├── fracture/
│   │   └── no_fracture/
│   │
│   └── test/
│       ├── fracture/
│       └── no_fracture/
│
├── models/
│   └── (stores model checkpoints, e.g., best_model.pth)
│
├── notebooks/
│   └── (stores Jupyter notebooks for research)
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── requirements.txt
```

---

## 1. Getting Started

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (optional, but highly recommended for fast training; defaults to CPU if unavailable)

### Installation
Clone or navigate to the repository directory, then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Preparation
To train the model, place your 2D images (PNG, JPG, or BMP formats) into the corresponding class directories under `dataset/`:

1. **Training Data**: `dataset/train/fracture/` and `dataset/train/no_fracture/`
2. **Validation Data**: `dataset/validation/fracture/` and `dataset/validation/no_fracture/`
3. **Test Data**: `dataset/test/fracture/` and `dataset/test/no_fracture/`

---

## 3. Running the Machine Learning Pipeline

### Step A: Model Training
Train a pre-trained ResNet18 model using transfer learning. The training script will optimize model parameters, validate progress after each epoch, save the best weights to `models/best_model.pth`, and export a training curve plot to `models/training_curves.png`.

```bash
python src/train.py --epochs 10 --batch_size 16 --lr 0.001
```

**Parameters:**
- `--epochs`: Number of learning epochs (default: `5`).
- `--batch_size`: Batch size (default: `8`).
- `--lr`: Learning rate (default: `0.001`).
- `--dataset_dir`: Path to dataset root (default: `dataset`).
- `--model_save_path`: Destination path for saved weights (default: `models/best_model.pth`).

### Step B: Evaluation
Assess the model's classification performance on the unseen `test` dataset. This script computes overall accuracy, classification metrics (Precision, Recall, F1-Score) for each class, prints a confusion matrix, and saves a visual confusion matrix diagram to `models/confusion_matrix.png`.

```bash
python src/evaluate.py
```

### Step C: Command-Line Inference
Test predictions on a single image file from the command line:

```bash
python src/predict.py --image dataset/test/fracture/fracture_0.png
```

---

## 4. Launching the Web Application

We provide a lightweight web server serving both the REST API endpoints and the diagnostic web dashboard.

### Start the Server
Start the Uvicorn application from the project root:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### Accessing the Web Dashboard
Open your browser and navigate to:
[**http://127.0.0.1:8000**](http://127.0.0.1:8000)

**Dashboard Features:**
- **System Indicator**: Real-time validation if the system is ready (checks if the model weights file `models/best_model.pth` exists and is loaded successfully).
- **Drag-and-Drop Uploader**: Upload cervical spine scans and view immediate interactive image previews.
- **Real-Time Classification**: Click **Analyze Scan** to post the scan to `/api/predict` and display classification outcomes (Fracture vs. No Fracture) alongside individual confidence percentage bars.
