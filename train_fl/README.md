# Hierarchical Multimodal Model for HR-NMIBC Prognosis - Training Pipeline

This guide outlines the steps to prepare your data, extract features, and train models for the Hierarchical Multimodal Model for HR-NMIBC Prognosis.

---

## 1. Configure Data Root
Set the data root path in `dataset_iterator.py` to point to your raw data directory.

---

## 2. Generate Data Splits
Create stratified k-fold cross-validation splits:

```bash
python create_split.py
```
- This will generate k CSV files for train, validation, and test sets in the `splits/` directory

---

## 3. Extract Features & Coordinates
Run feature extraction:

```bash
python extract_features.py
```

- Raw data structure:
```bash
data/
  └── {patient_id}/
      ├── {patient_id_CD.json}
      ├── {patient_id_HE.tif}
      └── {patient_id_HE_mask.tif}
```

- Model: [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
- Weights: Place model weights at `data/pytorch_model.bin`

---

## 4. Train the Model
Train using `train.py`:
```bash
python train.py --data_root_dir <features_dir> --split_dir <splits_dir> --exp_code <experiment_name>
```
- Parameters:
  - `--data_root_dir`: Directory containing extracted features
  - `--split_dir`: Directory containing split CSV files
  - `--exp_code`: Name for your experiment (any string)

---

For more details, see the documentation or contact the project maintainers.