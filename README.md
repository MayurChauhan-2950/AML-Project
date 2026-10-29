# Brain Tumor MRI Multi-Class Classification System
### Applied Machine Learning & Deep Learning Capstone Project

This project provides an end-to-end, academic-grade Deep Learning pipeline for classifying brain MRI scans into four distinct diagnostic categories:
1. **Glioma:** Malignant tumors originating in glial cells, typically aggressive with infiltrating margins.
2. **Meningioma:** Tumors arising from meningeal membranes, typically slow-growing and requiring monitoring.
3. **Pituitary Tumor:** Abnormal growths on the pituitary gland affecting endocrine and vision systems.
4. **No Tumor:** Healthy brain MRI tissue with no oncological abnormalities.

---

## 🔬 Project Architecture & Methodology
* **Medical Image Processing:** Grayscale and RGB analysis, dynamic color space conversions, pixel distribution analysis, and $[0, 255] \to [0.0, 1.0]$ normalization.
* **Input Pipeline Optimization:** TensorFlow `image_dataset_from_directory` with deterministic 80/20 train/validation splits, in-graph GPU/CPU data augmentation (random horizontal flip, rotation, zoom), in-memory caching, and asynchronous `AUTOTUNE` prefetching.
* **Architectural Benchmarking:**
  * **Model 1: Custom Deep CNN Baseline** — Built with 3 convolutional stages, Batch Normalization, and `GlobalAveragePooling2D` (drastically reducing parameters to ~150k while avoiding dense flattening parameter explosion).
  * **Model 2: Pre-trained ResNet50 Transfer Learning** — Pre-trained on ImageNet with frozen residual backbone and a customized multi-class classification head (`GlobalAveragePooling2D` $\to$ `Dense(128)` $\to$ `BatchNormalization` $\to$ `Dropout(0.4)` $\to$ `Dense(4, Softmax)`).
* **Comprehensive Clinical Evaluation:**
  * Dual-panel Training vs. Validation Learning Curves (Accuracy & Loss).
  * Multi-class Confusion Matrix Heatmaps.
  * Per-class Precision, Recall (Sensitivity), and Macro $F_1$-score.
  * One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) and Area Under Curve (AUC).
  * End-to-end Computer-Aided Diagnosis (CAD) clinical inference simulation.

---

## 💾 Dataset Access & Download

The MRI dataset used in this project is sourced from Kaggle:
* **Dataset Link:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Due to size constraints, the raw image dataset is not included directly in this repository. To run or reproduce the training pipeline from scratch:
1. Download the dataset from the Kaggle link above (or use `kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset`).
2. Extract the downloaded archive into a folder named `dataset/` in the project root:
   ```text
   Brain_Tumor_Classification_AML_Project/
   └── dataset/
       ├── Training/
       │   ├── glioma/
       │   ├── meningioma/
       │   ├── notumor/
       │   └── pituitary/
       └── Testing/
           ├── glioma/
           ├── meningioma/
           ├── notumor/
           └── pituitary/
   ```

---

## 🗂️ Project Directory Structure

```text
Brain_Tumor_Classification_AML_Project/
│
├── Brain_Tumor_Classification_AML.ipynb    # Main end-to-end Jupyter Notebook
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation & execution guide
├── model_comparison_results.csv            # Exported quantitative metric benchmark table
├── custom_cnn_model.keras                  # Saved Custom CNN weights & architecture
├── resnet50_model.keras                    # Saved ResNet50 Transfer Learning model
└── dataset/                                # Offline MRI image dataset
    ├── Training/                           # 5,600 training MRI scans
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/                            # 1,311 testing MRI scans
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

---

## ⚙️ Installation & Environment Setup

1. Open a terminal and navigate to the project directory:
   ```bash
   cd "Brain_Tumor_Classification_AML_Project"
   ```

2. Install all required dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

Dependencies include:
* `tensorflow>=2.16.0`
* `opencv-python>=4.9.0`
* `numpy>=1.26.0`
* `pandas>=2.0.0`
* `matplotlib>=3.8.0`
* `seaborn>=0.13.0`
* `scikit-learn>=1.4.0`
* `jupyter>=1.0.0`
* `ipykernel>=6.29.0`

*(Note: Download the dataset from the Kaggle link above and place it in `dataset/` before running the notebook.)*

---

## 🚀 How to Run the Project

### Option 1: Jupyter Lab / Jupyter Notebook
Launch Jupyter in the project directory:
```bash
jupyter notebook Brain_Tumor_Classification_AML.ipynb
```
Select the Python 3 kernel and run all cells sequentially (**Kernel > Restart & Run All**).

### Option 2: Command Line Execution via `nbconvert`
To execute the complete notebook and write outputs directly into the `.ipynb` file:
```bash
jupyter nbconvert --to notebook --execute Brain_Tumor_Classification_AML.ipynb --inplace
```

---

## 📊 Evaluation Outputs Generated
* **Dual-Panel Learning Curves:** Plots Training vs. Validation Loss and Accuracy across epochs.
* **Side-by-Side Confusion Matrices:** Multi-class heatmaps comparing Custom CNN vs. ResNet50 predictions.
* **Classification Reports:** Per-class Precision, Recall, and $F_1$-score breakdown.
* **One-vs-Rest (OvR) ROC Curves:** Showing per-class False Positive Rate vs. True Positive Rate and Area Under Curve (AUC).
* **Clinical CAD Inference Demonstration:** Interactive diagnostic function displaying an arbitrary test MRI scan with predicted classification, confidence percentage, and full probability distribution.
