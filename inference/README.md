# Hierarchical Multimodal Model for HR-NMIBC Prognosis - Inference Guide
This guide describes how to set up and run inference for the Hierarchical Multimodal Model for HR-NMIBC Prognosis.

---

## 1. Prepare Model Weights
- Copy the model weights for all folds into the folder:
```bash
model/<Model Name>/
```
- If you use a custom model name, update both the folder name and the corresponding name in `inference.py`.
- Download the UNI2-h model weights [from MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and place them in the model directory as:
```bash
model/pytorch_model.bin
```
---

## 2. Run Inference
- Use the provided bash scripts to execute inference.
- Ensure all required paths and parameters are set correctly in the scripts and `inference.py`.
- For further details, refer to the documentation or contact the project maintainers.

