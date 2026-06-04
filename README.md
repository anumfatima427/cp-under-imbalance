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
└── dfuc2021/
    ├── train/
    │   ├── train.csv
    │   └── images/
    └── test/
        ├── test.csv
        └── images/
```

