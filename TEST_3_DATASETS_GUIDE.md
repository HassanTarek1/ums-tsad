================================================================================
QUICK TEST GUIDE: Run UMS-TSAD Baseline on 3 Datasets
================================================================================

This guide shows you how to test the UMS-TSAD baseline framework on a small
testbed of just 3 datasets to verify everything works before running larger
experiments.

================================================================================
OPTION 1: USE THE AUTOMATED TEST SCRIPT (EASIEST)
================================================================================

Step 1: Review the test dataset list
-------------------------------------
The file test_3_datasets.csv contains 3 UCR Anomaly Archive datasets:
  001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt
  002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt
  003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt

Step 2: Update paths in test_3_datasets.sh (if needed)
-------------------------------------------------------
Open test_3_datasets.sh and verify these paths:

  DATASET_PATH="/home/maxoud/local-storage/projects/RAMSeS/Mononito/datasets"
  TRAINED_MODELS_PATH="/home/maxoud/local-storage/projects/RAMSeS/Mononito/trained_models"

Step 3: Run the test
--------------------
cd /home/maxoud/local-storage/projects/ums-tsad
./test_3_datasets.sh

Expected runtime: 30-90 seconds (10-30 seconds per dataset)

Step 4: Check results
---------------------
Results are saved to: test_results_3datasets_<timestamp>/
  - summary_report.txt   # Human-readable summary
  - results.csv          # Tabular results
  - detailed_results.json # Full details

================================================================================
OPTION 2: MANUAL COMMAND (MORE CONTROL)
================================================================================

cd /home/maxoud/local-storage/projects/ums-tsad

python3 run_testbed_baseline.py \
    --dataset_list test_3_datasets.csv \
    --trained_model_path /path/to/RAMSeS/Mononito/trained_models \
    --dataset_path /path/to/RAMSeS/Mononito/datasets \
    --output_dir test_results_3datasets \
    --downsampling 10 \
    --min_length 256 \
    --timeout 600

Parameters explained:
  --dataset_list       : CSV with 3 datasets to test
  --trained_model_path : Where your trained models are stored
  --dataset_path       : Root directory of datasets
  --output_dir         : Where to save results
  --downsampling 10    : Reduce data by factor of 10
  --min_length 256     : Minimum sequence length
  --timeout 600        : 10 minute timeout per dataset

================================================================================
OPTION 3: PYTHON SCRIPT (PROGRAMMATIC)
================================================================================

Create a file test_3.py:

```python
from run_testbed_baseline import UMSTSADTestbedRunner

runner = UMSTSADTestbedRunner(
    dataset_list_file='test_3_datasets.csv',
    trained_model_path='/path/to/trained_models',
    dataset_path='/path/to/datasets',
    output_base_dir='test_results_3datasets',
    timeout=600,
    downsampling=10,
    min_length=256
)

# Run on first 3 datasets (or all if CSV has exactly 3)
results_df = runner.run_testbed(max_datasets=3)

print(f"\\nProcessed {len(results_df)} datasets")
print(f"Success rate: {results_df['success'].mean()*100:.1f}%")
```

Run it:
```bash
python3 test_3.py
```

================================================================================
EXPECTED OUTPUT
================================================================================

During execution, you'll see:

================================================================================
Processing: 001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt (Domain: anomaly_archive)
================================================================================

Stage 1: Initializing RankModels...
Initialization complete: 0.52s

Stage 2: Evaluating models with 3 criteria...
Model evaluation complete: 8.34s

Stage 3: Ranking models...
Model ranking complete: 0.15s

Best Model (PR-AUC): LSTMVAE_1 (Score: 0.8234)
Best Model (F1): NN_1 (Score: 0.7891)

End-to-End Time: 9.01s
Peak Memory: 1245.67 MB

Progress: 1/3 datasets processed

[... similar output for datasets 2 and 3 ...]

================================================================================
RESULTS SUMMARY
================================================================================

After completion, check summary_report.txt:

================================================================================
UMS-TSAD Baseline Testbed Results Summary
Timestamp: 2025-12-29 14:30:00
================================================================================

OVERALL STATISTICS
--------------------------------------------------------------------------------
Total Datasets: 3
Successful: 3
Failed: 0

COMPUTATIONAL OVERHEAD
--------------------------------------------------------------------------------
Average Initialization Time: 0.45s (±0.05s)
Average Model Evaluation Time: 8.20s (±0.80s)
Average Model Ranking Time: 0.18s (±0.02s)
Average End-to-End Time: 8.83s (±0.85s)
Total Computational Time: 26.49s

MEMORY USAGE
--------------------------------------------------------------------------------
Average Peak Memory: 1234.56 MB
Max Peak Memory: 1456.78 MB
Average Memory: 987.65 MB

PERFORMANCE METRICS
--------------------------------------------------------------------------------
Average F1 Score: 0.7845 (±0.0234)
Median F1 Score: 0.7891
Average PR-AUC: 0.8123 (±0.0312)
Median PR-AUC: 0.8234

================================================================================
TROUBLESHOOTING
================================================================================

Issue: "Dataset not found"
---------------------------
Check that dataset files exist:
  ls /path/to/datasets/anomaly_archive/001_UCR_Anomaly_*.txt

Issue: "Trained models not found"
----------------------------------
Check trained model structure:
  ls /path/to/trained_models/anomaly_archive/001_UCR_Anomaly_*/

Expected structure:
  trained_models/
  └── anomaly_archive/
      └── 001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620/
          ├── LSTMVAE_1.pth
          ├── DGHL_1.pth
          ├── NN_1.pth
          ├── RNN_1.pth
          ├── LOF_1.pth
          ├── MD_1.pth
          └── CBLOF_1.pth

Issue: "Module not found"
--------------------------
Make sure you're in the ums-tsad directory:
  cd /home/maxoud/local-storage/projects/ums-tsad

Issue: Import errors
--------------------
Check Python environment:
  python3 test_testbed_setup.py

================================================================================
VERIFY RESULTS ARE REASONABLE
================================================================================

Sanity checks:
1. All 3 datasets completed successfully
2. End-to-end time: 5-15 seconds per dataset
3. Peak memory: 500-2000 MB per dataset
4. F1 scores: 0.3-0.9 (depends on dataset difficulty)
5. PR-AUC scores: 0.4-0.95 (depends on dataset)

If results look good, you're ready for the full testbed!

================================================================================
NEXT STEPS
================================================================================

Once the 3-dataset test works:

1. ✓ Verify results look reasonable
2. ✓ Check that all output files are generated
3. → Run on larger testbed:
   
   # 10 datasets
   ./run_baseline_testbed.sh ucr_sample
   
   # Full UCR (250 datasets)
   ./run_baseline_testbed.sh ucr_full
   
   # SMD dataset
   ./run_baseline_testbed.sh smd
   
   # SKAB dataset
   ./run_baseline_testbed.sh skab

4. → Compare with RAMSeS results:
   
   python3 compare_baseline_ramses.py \
       --baseline_dir test_results_3datasets \
       --ramses_dir /path/to/ramses_results

================================================================================
CUSTOMIZING THE TEST
================================================================================

To test different datasets, edit test_3_datasets.csv:

For UCR Anomaly Archive:
```csv
file_name,domain_name
010_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature6_4000_6006_6054.txt,anomaly_archive
020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2_5000_1902_1946.txt,anomaly_archive
030_UCR_Anomaly_DISTORTEDNoisyShocksensorData_10000_12496_12504.txt,anomaly_archive
```

For SMD:
```csv
file_name,domain_name
machine-1-1,SMD
machine-1-2,SMD
machine-1-3,SMD
```

For SKAB:
```csv
file_name,domain_name
anomaly-free-anomaly-free,SKAB
valve1-other,SKAB
valve2-other,SKAB
```

Note: Make sure the domain_name matches your dataset structure!

================================================================================
TIMING ESTIMATES
================================================================================

Per dataset (approximate):
- Initialization: 0.5-1 seconds
- Model Evaluation: 5-20 seconds (depends on dataset size)
- Model Ranking: 0.1-0.5 seconds
- Total: 6-22 seconds per dataset

For 3 datasets:
- Expected total time: 20-70 seconds
- With overhead: 30-90 seconds

For 10 datasets (ucr_sample):
- Expected total time: 60-220 seconds (~1-4 minutes)

For 250 datasets (ucr_full):
- Expected total time: 1500-5500 seconds (~25-90 minutes)

================================================================================
SUCCESS INDICATORS
================================================================================

✓ All 3 datasets show "success": True in results.csv
✓ summary_report.txt is generated
✓ detailed_results.json contains full metrics
✓ No Python errors in terminal output
✓ Timing and memory metrics are reasonable
✓ F1 and PR-AUC scores are in valid range (0-1)

If all checks pass: Ready to run full testbed! 🚀

================================================================================
