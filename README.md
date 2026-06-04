This is the code release accompanying the paper 'Conformal Prediction Under Class Imbalance for Medical Image Classification'

<img width="1920" height="1080" alt="methodology" src="https://github.com/user-attachments/assets/b54fe349-591f-49a9-853a-aac94cdfc7b2" />


The conformal prediction post-hoc analysis code is taken from this paper: [Class-Conditional Conformal Prediction with Many Classes](https://dl.acm.org/doi/10.5555/3666122.3668939)

## Setup:
First, create a virtual environment and install the necessary packages by running

 ```bash
conda create --name env
conda activate env
pip install -r requirements.txt
```

## Data Description

| Dataset | Classes | Train | Test | Minority Class |
|---------|---------|-------|------|----------------|
| **ISIC2019** (Skin) | 8 | 25,331 | 6,191 | Dermatofibroma (0.94%)|
| **DFUC2021** (Diabetic Foot Ulcers) | 4 | 5,955 | 5,734 | Ischaemia (3.9%) |
| **DDR** (Diabetic Retinopathy) | 5 | 10,017 | 2,505 | Severe (1.9%) |

### 1. Prepare Data

 ```bash
python 1-Data_Prep.py --data_dir /path/to/dfuc2021 --output_dir ./dataset_split --symlink
```

Organises each dataset as:
```
data/
├── isic2019/
│   ├── train/
│   │   ├── train.csv       # Columns: image, label, label_name
│   │   └── images/
│   └── test/
│       ├── test.csv
│       └── images/
```

### 3. Train Models

```bash
# Cross-Entropy baseline
python 2a-Train-CE.py --data_dir ./dataset_split --output ./model_output/baseline_ce --epochs 50 --batch_size 32

# Focal Loss baseline
python 2b-Train-Focal.py --data_dir ./dataset_split --output ./model_output/focal_loss --gamma 2.0

# Conformal-Aware Loss (ours)
python 2c-Train-CA.py \
    --train_csv ./train.csv \
    --test_csv ./test.csv \
    --train_dir ./images/train \
    --test_dir ./images/test \
    --output ./model_output/conformal
```

### 4. Extract Softmax Probabilities

```bash
python 3-Generate_Prob.py \
    --test_csv ./test.csv \
    --test_dir ./images/test \
    --model_path ./model_output/baseline_ce/best_model.pth \
    --output_npz ./[dataset]-cp-aware.npz \
    --temperature 1.5

```
### 5. Run Conformal Prediction Experiments

```bash
python calibration/run_experiments.py \
    --softmax_dir results/ddr/ca/softmax \
    --output_dir results/ddr/ca/conformal \
    --seeds 0 1 2 3 4 \
    --n_avg 50 100 150 
```

### 6. Generate Figures and Tables

```bash
python calibration/notebooks/create_latex_table.py 
python calibration/notebooks/generate_plots.py 
```
> **Note:** We evaluate Standard (STD), Conditional (CC), and Cluster-Wise (CW) calibration across varying calibration sizes ($n_{avg}$). Our Conformal-Aware (CA) loss uniquely enables the CW method to achieve CC-level safety (CovGap) while maintaining highly efficient prediction sets (AvgSize). Best results per block are in **bold**.

| Dataset | Method | Loss | CovGap↓ (n=50) | AvgSize↓ (n=50) | CovGap↓ (n=100) | AvgSize↓ (n=100) | CovGap↓ (n=150) | AvgSize↓ (n=150) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISIC2019** | **STD** | CE | 20.3 (0.6) | 2.1 (0.1) | 19.1 (0.8) | 2.2 (0.1) | 19.2 (0.5) | 2.2 (0.0) |
| | | Focal | 17.2 (0.7) | **1.9 (0.1)** | 16.6 (0.7) | **2.0 (0.1)** | 16.9 (0.6) | **2.0 (0.0)** |
| | | CA (Ours) | **15.3 (0.7)** | 2.1 (0.1) | **15.5 (0.6)** | 2.1 (0.1) | **15.7 (0.2)** | 2.1 (0.0) |
| | **CC** | CE | 5.8 (0.4) | 4.8 (0.3) | 5.7 (1.1) | 4.1 (0.3) | 3.9 (0.4) | 4.1 (0.2) |
| | | Focal | 5.8 (0.4) | 4.7 (0.3) | 4.2 (0.3) | **3.7 (0.3)** | 3.6 (0.5) | **3.5 (0.2)** |
| | | CA (Ours) | **5.0 (0.3)** | **4.7 (0.1)** | **3.5 (0.6)** | **3.7 (0.3)** | **3.2 (0.7)** | **3.5 (0.2)** |
| | **CW** | CE | 15.0 (0.4) | 2.6 (0.1) | 9.3 (0.7) | 2.8 (0.1) | 8.4 (1.2) | 3.1 (0.3) |
| | | Focal | 13.8 (0.6) | **2.4 (0.2)** | 8.4 (0.7) | **2.7 (0.2)** | 6.1 (1.4) | **2.9 (0.2)** |
| | | CA (Ours) | **11.2 (0.3)** | 2.6 (0.1) | **7.9 (1.0)** | 2.8 (0.2) | **6.2 (0.7)** | 3.0 (0.3) |
| | | | | | | | | |
| **DFUC2021** | **STD** | CE | 8.2 (0.7) | **1.9 (0.1)** | 8.5 (0.4) | **1.9 (0.0)** | 8.7 (0.4) | **1.9 (0.0)** |
| | | Focal | 9.6 (0.5) | 2.0 (0.0) | 10.2 (0.6) | 2.0 (0.0) | 10.6 (0.6) | 2.0 (0.0) |
| | | CA (Ours) | **5.2 (0.4)** | **1.9 (0.0)** | **5.3 (0.2)** | **1.9 (0.0)** | **5.5 (0.2)** | **1.9 (0.0)** |
| | **CC** | CE | 4.8 (0.3) | 3.1 (0.1) | 3.3 (0.4) | 2.4 (0.1) | 2.9 (0.5) | 2.2 (0.1) |
| | | Focal | **4.5 (0.6)** | 3.0 (0.1) | 3.8 (0.3) | 2.3 (0.0) | 3.2 (0.4) | 2.2 (0.1) |
| | | CA (Ours) | 5.1 (0.4) | **2.9 (0.1)** | **2.7 (0.3)** | **2.2 (0.1)** | **2.1 (0.2)** | **2.1 (0.0)** |
| | **CW** | CE | 8.0 (0.9) | 1.9 (0.1) | 6.9 (0.7) | **1.9 (0.0)** | 7.0 (0.9) | 1.9 (0.1) |
| | | Focal | 7.5 (0.9) | 2.1 (0.0) | 8.3 (0.9) | **1.9 (0.0)** | 8.2 (1.3) | 2.0 (0.0) |
| | | CA (Ours) | **5.1 (0.8)** | **1.9 (0.0)** | **3.8 (0.5)** | 2.0 (0.0) | **4.9 (0.7)** | **1.9 (0.0)** |
| | | | | | | | | |
| **DDR** | **STD** | CE | 17.9 (0.9) | **1.4 (0.0)** | 17.6 (0.4) | **1.4 (0.0)** | 17.1 (0.9) | **1.4 (0.0)** |
| | | Focal | 12.9 (1.2) | 1.6 (0.1) | 13.6 (1.4) | 1.5 (0.1) | 13.7 (1.5) | 1.5 (0.0) |
| | | CA (Ours) | **8.7 (0.5)** | **1.4 (0.0)** | **8.5 (0.5)** | **1.4 (0.0)** | **8.8 (0.5)** | **1.4 (0.0)** |
| | **CC** | CE | 5.2 (0.3) | 4.1 (0.3) | **3.9 (0.7)** | 2.8 (0.2) | 3.6 (0.6) | 2.6 (0.1) |
| | | Focal | 5.5 (0.4) | 3.0 (0.1) | 4.8 (0.4) | 2.9 (0.1) | 4.2 (0.6) | 2.4 (0.1) |
| | | CA (Ours) | **4.7 (0.4)** | **2.9 (0.1)** | **3.9 (0.1)** | **2.5 (0.1)** | **3.1 (0.4)** | **2.2 (0.1)** |
| | **CW** | CE | 16.4 (2.0) | **1.6 (0.1)** | 14.0 (1.6) | 1.7 (0.1) | 11.9 (2.1) | **1.7 (0.1)** |
| | | Focal | 11.0 (1.8) | 1.8 (0.2) | 10.6 (1.4) | 1.9 (0.1) | 9.1 (0.9) | 1.8 (0.0) |
| | | CA (Ours) | **9.2 (1.7)** | **1.6 (0.1)** | **7.6 (0.5)** | **1.6 (0.1)** | **6.5 (0.7)** | 1.7 (0.2) |
<img width="1791" height="601" alt="post-hoc_calib_results" src="https://github.com/user-attachments/assets/3224e550-e4ed-4e6d-a737-6a5cbbd7ca88" />

<img width="1026" height="528" alt="post_hoc_all_datasets" src="https://github.com/user-attachments/assets/ea519078-bd25-417d-85f0-368dbbfed55e" />



