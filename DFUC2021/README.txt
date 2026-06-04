python 0-Dataset_Characteristics.py

python 1-Data_Prep.py --data_dir /path/to/dfuc2021 --output_dir ./dataset_split --symlink


python 2a-Train-DFUC2021-CE.py --data_dir ./dataset_split --output ./model_output/baseline_ce --epochs 50 --batch_size 32

python 2b-Train-DFUC2021-Focal.py --data_dir ./dataset_split --output ./model_output/focal_loss --gamma 2.0

python 2c-Train-DFUC-CA.py \
    --train_csv ./train.csv \
    --test_csv ./test.csv \
    --train_dir ./images/train \
    --test_dir ./images/test \
    --output ./model_output/conformal

python 3-Generate_Prob.py \
    --test_csv ./test.csv \
    --test_dir ./images/test \
    --model_path ./model_output/baseline_ce/best_model.pth \
    --output_npz ./ddr-cp-aware.npz \
    --temperature 1.5
