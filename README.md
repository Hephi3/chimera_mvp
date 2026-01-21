# Hierarchical Multimodal Model for HR-NMIBC Prognosis
**CHIMERA Challenge Submission – Task 2**

Author: Philipp Hempel

---

## Overview
This repository contains code and instructions for training and inference of hierarchical multimodal models for HR-NMIBC prognosis, as submitted to the CHIMERA Challenge (Task 2).

- `train_mm_only/`: All scripts and files required for training the models. Place your data and UNI2-h model weights in the specified directories.
- `inference/`: All scripts and files required for running inference on the test set. Place the trained model weights and UNI2-h model weights in the specified directories.

For detailed setup and usage instructions, please refer to the `README.md` files in the respective folders.

- `test/`: This folder allows to test the multimodal inference model.

---

Furthermore it contains the additional scripts used for the extension for federated learning, and the CF & CD experiments.
- `train_fl\`: All scripts and files required for training the federated learning models using Flower.
- `train_cdcf/`: All scripts and files required for training the models for the CF & CD experiments with label distribution shift is used for the cd and domain shift is used for the cf. (-> Old setting)
- `train_cfcd/`: All scripts and files required for training the models for the CF & CD experiments with domain shift is used for the cd and label distribution shift is used for the cf. (-> New setting)