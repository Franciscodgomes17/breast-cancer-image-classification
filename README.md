# Breast Cancer Image Classification with Deep Learning

This project applies **Convolutional Neural Networks (CNNs)** to the task of **breast cancer diagnosis** using medical imaging data.  
It compares **binary classification** (benign vs malignant) with **multi-class classification** (cancer subtypes), evaluating models through multiple metrics and visualizations.

---

## Project Structure
- `Functions_Group_8.py` → Helper functions for preprocessing, training, and evaluation  
- `Notebook_Group_8.ipynb` → Jupyter notebook with experiments and results  
- `Report_Group_8.pdf` → Final academic report (with methodology, results, and discussion)  
- `Report_Group_8 - Latex/` → LaTeX source files for the report  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/breast-cancer-image-classification.git
cd breast-cancer-image-classification
pip install -r requirements.txt
```

---

## Requirements
Main dependencies include:

- numpy  
- pandas  
- scikit-learn  
- imbalanced-learn  
- Pillow  
- opencv-python  
- tensorflow  
- keras  
- matplotlib  
- seaborn  
- plotly  
- jupyter  

(See [`requirements.txt`](requirements.txt) for the full list)

---

## Features
- Preprocessing and augmentation of breast cancer imaging datasets  
- Binary classification (benign vs malignant)  
- Multi-class classification (cancer subtypes)  
- Deep Learning models based on CNNs:  
  - VGG16  
  - ResNet50  
  - Custom CNN  
- Evaluation metrics:  
  - ROC, Precision-Recall, F1 Score  
  - Confusion Matrices & Heatmaps  
  - Calibration curves  
- Comparative analysis between binary and multi-class approaches  

---

## Usage
Run the experiments by opening the Jupyter notebook:

```bash
jupyter notebook Notebook_Group_8.ipynb
```

Or import helper functions directly from `Functions_Group_8.py`.

---

## Report
The full research report is available here:  
👉 [Report_Group_8.pdf](Report_Group_8.pdf)

---

## Authors
- Carolina Almeida  
- Duarte Carvalho  
- Francisco Gomes  
- Margarida Henriques  
- Marta Monteiro  

*Group 8 - Machine Learning Project*

---

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
