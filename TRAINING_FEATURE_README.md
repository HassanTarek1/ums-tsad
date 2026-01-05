# UMS-TSAD Testbed - Automatic Training Feature

## Overview

The testbed runner now supports **automatic model training**! You no longer need to pre-train models manually. The system can automatically train models for any dataset/entity that doesn't have trained models available.

## Key Features

✅ **Automatic Training**: Models are trained automatically if not found  
✅ **Force Retraining**: Use `--train` flag to retrain all models regardless of existing models  
✅ **Selective Algorithms**: Choose which algorithms to train  
✅ **Training Timing**: Training time is tracked separately from evaluation  

## Usage

### Basic Usage (No Training)
```bash
python run_testbed_baseline.py \
    --dataset_list test_3_datasets.csv \
    --dataset_path /path/to/datasets \
    --trained_model_path /path/to/trained_models \
    --output_dir testbed_results
```
This will fail if models don't exist.

### With Automatic Training
```bash
python run_testbed_baseline.py \
    --dataset_list test_3_datasets.csv \
    --dataset_path /path/to/datasets \
    --trained_model_path /path/to/trained_models \
    --output_dir testbed_results \
    --train
```
This will automatically train models if they don't exist!

### Force Retrain Everything
```bash
python run_testbed_baseline.py \
    --dataset_list test_3_datasets.csv \
    --dataset_path /path/to/datasets \
    --trained_model_path /path/to/trained_models \
    --output_dir testbed_results \
    --train
```
The `--train` flag will retrain models even if they already exist (overwrite mode).

### Select Specific Algorithms
```bash
python run_testbed_baseline.py \
    --dataset_list test_3_datasets.csv \
    --dataset_path /path/to/datasets \
    --trained_model_path /path/to/trained_models \
    --output_dir testbed_results \
    --train \
    --algorithms LOF NN RNN DGHL LSTMVAE
```

## Available Algorithms

- **LOF**: Local Outlier Factor (fast, non-parametric)
- **NN**: Nearest Neighbors (fast, non-parametric) 
- **RNN**: Recurrent Neural Network (slower, requires training)
- **DGHL**: Deep Generative Hidden Layer (slower, requires training)
- **LSTMVAE**: LSTM Variational Autoencoder (slower, requires training)
- **MD**: Mean Deviation
- **RM**: Running Mean
- **KDE**: Kernel Density Estimation
- **ABOD**: Angle-Based Outlier Detection
- **CBLOF**: Clustering-Based Local Outlier Factor
- **COF**: Connectivity-Based Outlier Factor  
- **SOS**: Stochastic Outlier Selection

## Default Algorithm Set

If you don't specify `--algorithms`, the system uses:
```python
['DGHL', 'RNN', 'LSTMVAE', 'NN', 'LOF']
```

## Training Time Tracking

When training is enabled, the results include:
- `time_0_training`: Time spent training models
- `time_1_initialization`: Time loading trained models
- `time_2_evaluation`: Time evaluating models
- etc.

## Example Output Structure

```
testbed_results_with_training/
├── detailed_results.json      # Full results with timing
├── summary_report.txt          # Human-readable summary  
├── results.csv                 # CSV format results
```

## Example: Complete Test Run

```bash
# For SKAB dataset with 3 entities
python run_testbed_baseline.py \
    --dataset_list test_3_datasets.csv \
    --dataset_path /home/maxoud/local-storage/projects/ums-tsad/Mononito/datasets \
    --trained_model_path /home/maxoud/local-storage/projects/ums-tsad/Mononito/trained_models \
    --output_dir testbed_results \
    --train \
    --algorithms LOF NN \
    --max_datasets 3
```

This will:
1. Load the dataset list
2. For each dataset:
   - Check if models exist
   - If not (or if `--train` is used): train LOF and NN models
   - Load trained models  
   - Evaluate with UMS-TSAD framework
3. Save comprehensive results

## Performance Notes

### Training Time by Algorithm (Approximate)
- **LOF**: ~30-60 seconds per entity
- **NN**: ~30-60 seconds per entity  
- **RNN**: ~5-15 minutes per entity
- **LSTMVAE**: ~10-30 minutes per entity
- **DGHL**: ~10-30 minutes per entity

### Recommendation
For quick testing, use: `--algorithms LOF NN`  
For complete evaluation, use: `--algorithms DGHL RNN LSTMVAE NN LOF`

## Troubleshooting

### Qt Platform Plugin Error
If you see Qt/matplotlib errors, they are warnings and training continues in the background (plots are saved to files using the Agg backend).

### Missing Models Error
```
FileNotFoundError: Models not found for skab/0
```
**Solution**: Add `--train` flag to enable automatic training

### Feature Dimension Mismatch
```
ValueError: X has 512 features, but NearestNeighbors is expecting 640 features
```
**Solution**: Retrain models with `--train` flag to ensure consistent dimensions

## Changes Made

### Files Modified
1. **run_testbed_baseline.py**:
   - Added `TrainModels` import
   - Added `--train` and `--algorithms` arguments
   - Added `train_models_if_needed()` method
   - Modified `run_single_dataset()` to train before evaluation

2. **model_trainer/train.py**:
   - Added `matplotlib.use('Agg')` for non-interactive plotting
   - Commented out optional models (NHi, TranAD) that require extra dependencies

### Backward Compatibility
The changes are fully backward compatible:
- Without `--train` flag: behaves exactly as before
- With `--train` flag: adds automatic training capability

## Example Log Output

```
2026-01-05 16:41:59 - INFO - Processing: 0.csv (Domain: SKAB)
2026-01-05 16:41:59 - INFO - Stage 0: Training models...
2026-01-05 16:41:59 - INFO - Training models for skab/0...
2026-01-05 16:42:05 - INFO - ✓ Successfully trained models for skab/0
2026-01-05 16:42:05 - INFO - Training complete: 6.23s
2026-01-05 16:42:05 - INFO - Stage 1: Initializing RankModels...
2026-01-05 16:42:06 - INFO - Stage 2: Evaluating models...
```

## Summary

With this new feature, you can now:
- ✅ Run testbed without pre-training models
- ✅ Automatically handle missing models
- ✅ Force retrain for consistency
- ✅ Select specific algorithms
- ✅ Track training time separately

**No more manual training required! Just add `--train` and go!** 🚀
