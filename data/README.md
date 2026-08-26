# Data

This project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The models consume only the nine temporal signals under `train/Inertial Signals/` and `test/Inertial Signals/`:

- `body_acc_{x,y,z}`
- `body_gyro_{x,y,z}`
- `total_acc_{x,y,z}`

Each sample is represented as `[128, 9]`. The precomputed 561-feature files `X_train.txt` and `X_test.txt` are not used as model inputs.

Download the dataset with:

```bash
python scripts/download_data.py
```
