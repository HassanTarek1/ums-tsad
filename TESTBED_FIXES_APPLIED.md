================================================================================
UMS-TSAD BASELINE TESTBED - FIXES APPLIED
================================================================================

Date: December 29, 2025
Purpose: Document all fixes applied to avoid issues found in RAMSeS development

================================================================================
ISSUE #1: SKAB CASE SENSITIVITY
================================================================================

PROBLEM:
--------
In RAMSeS testbed, we discovered that dataset names must be lowercase for 
consistency with the load_data() function in Datasets/load.py.

The load_data() function does:
```python
dataset = dataset.lower()  # Line 50 in RAMSeS/Datasets/load.py
```

However, testbed file lists may have mixed case domain names:
- "SKAB" in CSV file
- "SMD" in CSV file  
- "anomaly_archive" in CSV file

If we pass "SKAB" directly, it would become "skab" in load_data(), which is
correct. However, for consistency and to match RAMSeS behavior, we should
normalize domain names early.

ROOT CAUSE:
-----------
Original code in run_testbed_baseline.py:
```python
if domain == 'SKAB':  # ❌ Case-sensitive comparison
    dataset_name = 'SKAB'  # ❌ Wrong case
```

This could cause issues if:
1. Testbed CSV has "skab" (lowercase) → comparison fails
2. Dataset name passed with wrong case → inconsistent behavior

FIX APPLIED:
------------
```python
# Normalize domain to lowercase for consistency with RAMSeS
domain_lower = domain.lower()

if domain_lower == 'skab':  # ✓ Case-insensitive
    dataset_name = 'skab'  # ✓ Correct case
```

LOCATION:
---------
File: run_testbed_baseline.py
Function: convert_entity_name()
Lines: 136-177

RELATED RAMSES FILES:
---------------------
- RAMSeS/Datasets/load.py (line 50: dataset = dataset.lower())
- RAMSeS/run_testbed_comprehensive.py (line 277: domain.lower())

================================================================================
ISSUE #2: MISSING TRACEBACK IMPORT
================================================================================

PROBLEM:
--------
The code used traceback.format_exc() for error logging but didn't import 
the traceback module.

ERROR MESSAGE:
--------------
```
NameError: name 'traceback' is not defined
```

FIX APPLIED:
------------
Added import statement:
```python
import traceback
```

LOCATION:
---------
File: run_testbed_baseline.py
Line: 12 (in imports section)

================================================================================
ISSUE #3: TRAINED MODEL PATH STRUCTURE
================================================================================

VERIFICATION:
-------------
The script correctly uses the trained_model_path structure expected by
UMS-TSAD's RankModels class:

Expected structure:
```
trained_model_path/
├── anomaly_archive/
│   └── 001_UCR_Anomaly_XXX/
│       ├── LSTMVAE_1.pth
│       ├── DGHL_1.pth
│       └── ...
├── smd/
│   └── machine-1-1/
│       └── ...
└── skab/
    └── entity_name/
        └── ...
```

The code constructs the path correctly:
```python
TRAINED_MODELS_PATH = os.path.join(trained_model_path, dataset, entity)
```

This matches RAMSeS structure where:
- dataset is lowercase (e.g., "skab", "smd", "anomaly_archive")
- entity is the full entity name

STATUS: ✓ No fix needed - already correct

================================================================================
ISSUE #4: PARALLEL EXECUTION RACE CONDITION
================================================================================

CONTEXT:
--------
RAMSeS had a race condition when running with --parallel true, where multiple
threads would share the same test_data object, leading to:

```
ERROR - X has 2432 features, but NearestNeighbors is expecting 2368 features
```

APPLICABILITY TO UMS-TSAD BASELINE:
------------------------------------
UMS-TSAD baseline runs SEQUENTIALLY (no parallel execution), so this issue
does NOT apply. The baseline:

1. Loads models one by one
2. Evaluates each model sequentially
3. No concurrent data access

STATUS: ✓ Not applicable - sequential execution only

================================================================================
ISSUE #6: SKAB TRAIN/TEST SPLIT (80/20)
================================================================================

CONTEXT FROM RAMSES:
--------------------
In RAMSeS, we discovered and fixed an issue where SKAB dataset splitting was
not done chronologically. The correct approach is:
- First 80% of data → Training set  
- Last 20% of data → Test set

This ensures temporal ordering is preserved (critical for time series).

VERIFICATION IN UMS-TSAD:
-------------------------
Checked the load_skab() function in datasets/load.py:

```python
# Line 314: Compute split point
split_idx = int(0.8 * n_time)

# Lines 326-328: Fit scaler on TRAINING data only
if normalize:
    scaler = MinMaxScaler()
    scaler.fit(X.iloc[:split_idx])  # ✓ Fit on first 80%
    X = scaler.transform(X)         # ✓ Apply to all

# Lines 341-348: Split by group
if group == "train":
    Y_split = Y[:, :split_idx]      # ✓ First 80%
    labels = y[:split_idx]
elif group == "test":
    Y_split = Y[:, split_idx:]      # ✓ Last 20%
    labels = y[split_idx:]
```

STATUS: ✓ ALREADY CORRECT - No fix needed!

The UMS-TSAD implementation already does chronological 80/20 splitting correctly:
1. Split point calculated correctly (line 314)
2. Scaler fit on training portion only (line 327)
3. Train uses first 80%, test uses last 20% (lines 342, 346)
4. Temporal ordering preserved
5. Normalization done correctly (fit on train, apply to all)

This matches the corrected RAMSeS implementation in Datasets/load.py.

CRITICAL IMPORTANCE:
--------------------
Why this matters:
- ❌ Random split: Leaks future information into training
- ❌ Test-then-train: Violates causality
- ✅ Chronological 80/20: Preserves temporal order, realistic evaluation

The UMS-TSAD baseline correctly implements this!

================================================================================
ISSUE #7: MEMORY MONITORING
================================================================================

BEST PRACTICE:
--------------
RAMSeS validates that datasets exist before processing. Added similar
validation to the baseline.

IMPLEMENTED:
------------
The testbed runner checks:
1. Dataset list file exists
2. Trained models directory exists
3. Dataset path is accessible

These checks happen in:
- test_testbed_setup.py (verification script)
- run_baseline_testbed.sh (bash wrapper with validation)

STATUS: ✓ Already implemented

================================================================================
ISSUE #8: MEMORY MONITORING
================================================================================

VERIFICATION:
-------------
The baseline includes proper memory monitoring:

```python
class MemoryMonitor:
    def update(self):
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.measurements.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
```

Called at key points:
- After initialization
- After model evaluation
- After model ranking

STATUS: ✓ Already correct

================================================================================
ISSUE #7: SKAB TRAIN/TEST SPLIT
================================================================================

CONTEXT FROM RAMSES:
--------------------
In RAMSeS, we discovered and fixed an issue where SKAB dataset splitting was
not done chronologically. The correct approach is:
- First 80% of data → Training set
- Last 20% of data → Test set

This ensures temporal ordering is preserved (critical for time series).

VERIFICATION IN UMS-TSAD:
-------------------------
Checked the load_skab() function in datasets/load.py:

```python
# Line 314: Compute split point
split_idx = int(0.8 * n_time)

# Lines 326-328: Fit scaler on TRAINING data only
if normalize:
    scaler = MinMaxScaler()
    scaler.fit(X.iloc[:split_idx])  # ✓ Fit on first 80%
    X = scaler.transform(X)         # ✓ Apply to all

# Lines 341-348: Split by group
if group == "train":
    Y_split = Y[:, :split_idx]      # ✓ First 80%
    labels = y[:split_idx]
elif group == "test":
    Y_split = Y[:, split_idx:]      # ✓ Last 20%
    labels = y[split_idx:]
```

STATUS: ✓ ALREADY CORRECT - No fix needed!

The UMS-TSAD implementation already does chronological 80/20 splitting correctly:
1. Split point calculated correctly
2. Scaler fit on training portion only
3. Train uses first 80%, test uses last 20%
4. Temporal ordering preserved

This matches the corrected RAMSeS implementation.

================================================================================
ISSUE #8: ERROR HANDLING
================================================================================

VERIFICATION:
-------------
The baseline includes comprehensive error handling:

```python
try:
    # Run evaluation
    ...
except Exception as e:
    logger.error(f"Error processing {file_name}: {str(e)}")
    logger.error(traceback.format_exc())
    
    return {
        'success': False,
        'error': str(e),
        'timing': timing_dict,
        ...
    }
```

This ensures:
1. Individual dataset failures don't crash entire testbed
2. Errors are logged with full stack trace
3. Partial results are still saved

STATUS: ✓ Already correct

================================================================================
COMPLETE LIST OF CHANGES
================================================================================

1. ✅ Added traceback import
   - File: run_testbed_baseline.py
   - Line: 12

2. ✅ Fixed SKAB case sensitivity
   - File: run_testbed_baseline.py
   - Function: convert_entity_name()
   - Change: Use domain.lower() for all comparisons
   - Lines: 161-177

3. ✅ Updated all dataset name handling
   - Changed: 'SKAB' → 'skab'
   - Changed: 'SMD' → 'smd'  
   - Changed: 'anomaly_archive' → lowercase comparison
   - Reason: Match RAMSeS load_data() behavior

4. ✅ Added inline comments
   - Explained why lowercase is used
   - Referenced RAMSeS consistency

================================================================================
TESTING RECOMMENDATIONS
================================================================================

1. TEST CASE SENSITIVITY:
   -------------------------
   Create a test CSV with mixed case domain names:
   ```csv
   file_name,domain_name
   test1.txt,SKAB
   test2.txt,skab
   test3.txt,Skab
   ```
   
   Expected: All should work identically

2. TEST ERROR HANDLING:
   ---------------------
   - Try with non-existent entity
   - Try with corrupted model file
   - Verify error is caught and logged

3. TEST MEMORY MONITORING:
   ------------------------
   - Verify peak memory is tracked
   - Check that measurements are collected at each stage

4. COMPARE WITH RAMSES:
   ---------------------
   Run same datasets through both frameworks and verify:
   - Both use same dataset names (lowercase)
   - Both load same entities
   - Results are comparable

================================================================================
COMPATIBILITY MATRIX
================================================================================

┌──────────────────────────────┬─────────────────┬──────────────────────┐
│ Feature                      │ UMS-TSAD Base   │ RAMSeS               │
├──────────────────────────────┼─────────────────┼──────────────────────┤
│ Dataset name normalization   │ ✓ (lowercase)   │ ✓ (lowercase)        │
│ SKAB 80/20 chronological     │ ✓ (correct)     │ ✓ (correct)          │
│ Trained model path structure │ ✓ (compatible)  │ ✓ (compatible)       │
│ Error handling               │ ✓ (try/except)  │ ✓ (try/except)       │
│ Memory monitoring            │ ✓ (psutil)      │ ✓ (psutil)           │
│ Parallel execution           │ ✗ (sequential)  │ ✓ (optional)         │
│ Case sensitivity             │ ✓ (fixed)       │ ✓ (correct)          │
└──────────────────────────────┴─────────────────┴──────────────────────┘

================================================================================
VERIFICATION CHECKLIST
================================================================================

Before running baseline testbed:

☑ Imports include traceback
☑ Domain names converted to lowercase
☑ Dataset names match RAMSeS format
☑ SKAB 80/20 chronological split verified correct
☑ Trained model path structure correct
☑ Error handling comprehensive
☑ Memory monitoring in place
☑ Test setup script passes all checks

Run verification:
```bash
cd /home/maxoud/local-storage/projects/ums-tsad
python3 test_testbed_setup.py
```

Expected output:
```
================================================================================
✓ All checks passed! Ready to run baseline testbed.
================================================================================
```

================================================================================
NEXT STEPS
================================================================================

1. ✅ All fixes applied
2. ✅ Code reviewed for consistency
3. ⏳ Run test with ucr_sample
4. ⏳ Verify results are reasonable
5. ⏳ Run full testbed
6. ⏳ Compare with RAMSeS results

Command to run:
```bash
./run_baseline_testbed.sh ucr_sample
```

================================================================================
SUMMARY
================================================================================

All known issues from RAMSeS development have been reviewed for the UMS-TSAD
baseline testbed runner:

✓ Case sensitivity fixed (SKAB → skab)
✓ Missing import added (traceback)
✓ SKAB 80/20 chronological split verified (already correct!)
✓ Error handling robust
✓ Memory monitoring working
✓ Dataset path handling correct
✓ Compatible with RAMSeS testbed format

KEY FINDING: The UMS-TSAD baseline already had the correct SKAB train/test
split implementation (first 80% for training, last 20% for testing) with
proper chronological ordering and normalization fitted only on training data.

The baseline is now ready to run and will produce results directly comparable
to RAMSeS for computational overhead analysis.

================================================================================
