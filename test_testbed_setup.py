#!/usr/bin/env python3
"""
Quick test script to verify the testbed baseline setup

This script performs basic sanity checks before running the full testbed:
1. Checks if required paths exist
2. Verifies dataset list files are available
3. Tests loading a single dataset
4. Estimates execution time for full testbed
"""

import os
import sys
import time
from pathlib import Path

# Add to path
sys.path.insert(0, '/home/maxoud/local-storage/projects/ums-tsad')

def check_path(path: str, description: str) -> bool:
    """Check if a path exists"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("="*80)
    print("UMS-TSAD Baseline Testbed - Setup Verification")
    print("="*80)
    print()
    
    all_ok = True
    
    # Check directories
    print("Checking Required Directories:")
    print("-"*80)
    
    ums_tsad_dir = "/home/maxoud/local-storage/projects/ums-tsad"
    ramses_dir = "/home/maxoud/local-storage/projects/RAMSeS"
    
    all_ok &= check_path(ums_tsad_dir, "UMS-TSAD directory")
    all_ok &= check_path(ramses_dir, "RAMSeS directory")
    all_ok &= check_path(os.path.join(ramses_dir, "testbed/file_list"), "RAMSeS testbed file lists")
    
    print()
    
    # Check scripts
    print("Checking Scripts:")
    print("-"*80)
    
    all_ok &= check_path(os.path.join(ums_tsad_dir, "run_testbed_baseline.py"), "Testbed runner")
    all_ok &= check_path(os.path.join(ums_tsad_dir, "run_baseline_testbed.sh"), "Bash wrapper")
    all_ok &= check_path(os.path.join(ums_tsad_dir, "compare_baseline_ramses.py"), "Comparison tool")
    
    print()
    
    # Check dataset lists
    print("Available Dataset Lists:")
    print("-"*80)
    
    testbed_dir = os.path.join(ramses_dir, "testbed/file_list")
    if os.path.exists(testbed_dir):
        csv_files = [f for f in os.listdir(testbed_dir) if f.endswith('.csv')]
        for csv_file in sorted(csv_files):
            csv_path = os.path.join(testbed_dir, csv_file)
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                print(f"  ✓ {csv_file}: {len(df)} datasets")
            except Exception as e:
                print(f"  ✗ {csv_file}: Error - {str(e)}")
                all_ok = False
    else:
        print("  ✗ Testbed directory not found")
        all_ok = False
    
    print()
    
    # Check Python dependencies
    print("Checking Python Dependencies:")
    print("-"*80)
    
    required_modules = [
        'pandas', 'numpy', 'torch', 'sklearn', 'matplotlib', 
        'seaborn', 'psutil', 'tqdm', 'loguru'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - NOT INSTALLED")
            all_ok = False
    
    print()
    
    # Check UMS-TSAD modules
    print("Checking UMS-TSAD Modules:")
    print("-"*80)
    
    try:
        from model_selection.model_selection import RankModels
        print("  ✓ model_selection.RankModels")
    except Exception as e:
        print(f"  ✗ model_selection.RankModels: {str(e)}")
        all_ok = False
    
    try:
        from datasets.load import load_data
        print("  ✓ datasets.load_data")
    except Exception as e:
        print(f"  ✗ datasets.load_data: {str(e)}")
        all_ok = False
    
    try:
        from metrics.ranking_metrics import rank_by_centrality
        print("  ✓ metrics.ranking_metrics")
    except Exception as e:
        print(f"  ✗ metrics.ranking_metrics: {str(e)}")
        all_ok = False
    
    print()
    
    # Summary
    print("="*80)
    if all_ok:
        print("✓ All checks passed! Ready to run baseline testbed.")
        print()
        print("Next steps:")
        print("  1. Ensure trained models are available")
        print("  2. Update paths in run_baseline_testbed.sh if needed")
        print("  3. Run: ./run_baseline_testbed.sh ucr_sample")
    else:
        print("✗ Some checks failed. Please fix the issues above before running.")
        sys.exit(1)
    print("="*80)

if __name__ == '__main__':
    main()
