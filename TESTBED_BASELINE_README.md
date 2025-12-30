# UMS-TSAD Baseline Testbed Runner

This module enables running the UMS-TSAD framework (the original paper implementation) over the same testbed as RAMSeS to measure end-to-end computational overhead and serve as a baseline comparison.

## Overview

**UMS-TSAD** ("Unsupervised Model Selection for Time-series Anomaly Detection") uses **3 main model selection criteria**:

1. **Centrality-based ranking** - Ranks models by their position in score space using k-nearest neighbors
2. **Synthetic anomaly injection** - Evaluates models on injected anomalies (extremum, shift, trend, variance)
3. **Forecasting metrics** - Ranks models by forecasting performance (MAE, MSE, SMAPE, MAPE, likelihood)

This serves as a baseline to compare against **RAMSeS**, which extends these criteria with:
- Ensemble optimization via Genetic Algorithm
- Linear Thompson Sampling for online selection
- GAN-based robustness testing
- Borderline/off-by-threshold sensitivity tests
- Monte Carlo noise stress-tests
- Markov-chain rank aggregation

## Files

- `run_testbed_baseline.py` - Main testbed runner script
- `run_baseline_testbed.sh` - Bash wrapper for easy execution
- `compare_baseline_ramses.py` - Comparison utility to analyze results against RAMSeS

## Requirements

```bash
# Ensure you have the ums-tsad dependencies installed
pip install -r requirements.txt

# Additional for plotting
pip install seaborn
```

## Quick Start

### 1. Setup Paths

Edit the paths in `run_baseline_testbed.sh`:

```bash
RAMSES_DIR="/home/maxoud/local-storage/projects/RAMSeS"
UMS_TSAD_DIR="/home/maxoud/local-storage/projects/ums-tsad"
DATASET_BASE_PATH="/path/to/Mononito/datasets"
TRAINED_MODELS_BASE="/path/to/trained_models"
```

### 2. Run Baseline Testbed

```bash
# Make script executable
chmod +x run_baseline_testbed.sh

# Run on small UCR sample (10 datasets) - RECOMMENDED for testing
./run_baseline_testbed.sh ucr_sample

# Run on full UCR Anomaly Archive (250 datasets)
./run_baseline_testbed.sh ucr_full

# Run on SMD dataset
./run_baseline_testbed.sh smd

# Run on SKAB dataset
./run_baseline_testbed.sh skab

# Limit to N datasets (for testing)
./run_baseline_testbed.sh ucr_sample 5
```

### 3. Check Results

Results are saved to `ums-tsad/testbed_results/<dataset>_<timestamp>/`:

```
testbed_results/ucr_sample_20251229_120000/
├── detailed_results.json    # Full detailed results
├── results.csv              # Tabular results for analysis
├── summary_report.txt       # Human-readable summary
└── testbed_run.log         # Execution log
```

### 4. Compare with RAMSeS

Once you have both baseline and RAMSeS results:

```bash
python3 compare_baseline_ramses.py \
    --baseline_dir ums-tsad/testbed_results/ucr_sample_20251229_120000 \
    --ramses_dir RAMSeS/testbed_results/ucr_sample_20251229_140000 \
    --output_dir comparison_results
```

This generates:
- `comparison_report.txt` - Detailed comparison report
- `overhead_comparison.png` - Computational overhead plots
- `performance_comparison.png` - F1/PR-AUC comparison plots
- `comparison_data.json` - Structured comparison data

## Manual Execution

If you prefer not to use the bash script:

```bash
cd /home/maxoud/local-storage/projects/ums-tsad

python3 run_testbed_baseline.py \
    --dataset_list /path/to/RAMSeS/testbed/file_list/ucr_sample_10.csv \
    --trained_model_path /path/to/trained_models \
    --dataset_path /path/to/Mononito/datasets \
    --output_dir testbed_results/my_run \
    --downsampling 10 \
    --min_length 256 \
    --max_datasets 10  # Optional: limit for testing
```

## Expected Computational Overhead

The UMS-TSAD baseline focuses on **3 core selection criteria**, making it more lightweight than RAMSeS:

| Stage | UMS-TSAD | RAMSeS |
|-------|----------|--------|
| Model Evaluation | ✓ | ✓ |
| Centrality Ranking | ✓ | ✗ |
| Synthetic Anomalies | ✓ | ✓ (enhanced) |
| Forecasting Metrics | ✓ | ✗ |
| **Ensemble GA** | **✗** | **✓** |
| **Thompson Sampling** | **✗** | **✓** |
| **GAN Robustness** | **✗** | **✓** |
| **Borderline Tests** | **✗** | **✓** |
| **Monte Carlo** | **✗** | **✓** |
| **Rank Aggregation** | ✗ | ✓ |

Expected timing per dataset:
- **UMS-TSAD Baseline**: ~5-15 seconds
- **RAMSeS**: ~30-90 seconds (depending on modules enabled)

The overhead factor is approximately **2-6x**, with the tradeoff being improved robustness and adaptability.

## Output Metrics

The testbed runner collects:

### Computational Overhead
- Initialization time
- Model evaluation time (centrality + synthetic + forecasting)
- Model ranking time
- End-to-end time
- Peak memory usage
- Average memory usage

### Performance Metrics
- Best model selected (by PR-AUC)
- Best model selected (by F1)
- PR-AUC score of best model
- F1 score of best model
- Number of models evaluated
- Full performance matrix with all criteria

## Troubleshooting

### Issue: "Dataset list file not found"
- Check that the `RAMSES_DIR` path is correct
- Ensure RAMSeS testbed file lists exist: `RAMSeS/testbed/file_list/*.csv`

### Issue: "Trained models not found"
- Verify `TRAINED_MODELS_BASE` path
- Ensure models are trained for the datasets you're testing
- Models should be organized as: `trained_models/<dataset>/<entity>/<model_name>.pth`

### Issue: "Module not found" errors
- Ensure you're running from the ums-tsad directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Issue: Dataset loading fails
- Check `DATASET_BASE_PATH` points to correct location
- Verify dataset structure matches expected format
- Check that `downsampling` and `min_length` parameters are appropriate

## Understanding the Results

### Summary Report Structure

```
OVERALL STATISTICS
- Total datasets processed
- Success/failure counts

COMPUTATIONAL OVERHEAD
- Average initialization time
- Average model evaluation time
- Average model ranking time
- Average end-to-end time
- Total computational time

MEMORY USAGE
- Average peak memory
- Max peak memory

PERFORMANCE METRICS
- Average F1 score across all datasets
- Average PR-AUC across all datasets

PER-DOMAIN STATISTICS
- Breakdown by dataset domain (UCR, SMD, SKAB, etc.)
```

### Comparison Report

When comparing with RAMSeS:
- **Computational Overhead**: Shows the relative overhead factor and additional time
- **Performance Improvement**: Shows whether RAMSeS improves detection accuracy
- **Cost-Benefit Analysis**: Helps determine if additional overhead is justified

## Citation

If you use this baseline comparison in your research:

```bibtex
@article{goswami2023unsupervised,
  title={Unsupervised Model Selection for Time-series Anomaly Detection},
  author={Goswami, Mononito and Challu, Cristian and Callot, Laurent and Minorics, Lenon and Kan, Andrey},
  journal={arXiv preprint arXiv:2210.01078},
  year={2023}
}
```

## Support

For issues specific to:
- **UMS-TSAD framework**: Check the original paper and implementation
- **Testbed integration**: Review this README and check paths/configurations
- **RAMSeS comparison**: Ensure both frameworks are run on the same testbed datasets

## Next Steps

1. ✅ Run baseline testbed on sample datasets
2. ✅ Verify results look reasonable
3. ✅ Run on full testbed (or specific domains of interest)
4. ✅ Run RAMSeS on same testbed
5. ✅ Use comparison tool to analyze results
6. 📊 Include in your paper/report as baseline comparison

Good luck with your experiments!
