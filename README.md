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




