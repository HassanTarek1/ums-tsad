#!/usr/bin/env python3
"""
Regenerate summary report from existing detailed results JSON
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

def regenerate_summary(results_dir):
    """Regenerate summary from detailed_results.json"""
    
    # Load detailed results
    detailed_file = os.path.join(results_dir, 'detailed_results.json')
    if not os.path.exists(detailed_file):
        print(f"Error: {detailed_file} not found")
        return
    
    with open(detailed_file, 'r') as f:
        results = json.load(f)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate summary
    summary_file = os.path.join(results_dir, 'summary_report.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UMS-TSAD Baseline Testbed Results Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Datasets: {len(results_df)}\n")
        f.write(f"Successful: {results_df['success'].sum()}\n")
        f.write(f"Failed: {(~results_df['success']).sum()}\n\n")
        
        # Successful results only
        success_df = results_df[results_df['success']]
        
        if len(success_df) > 0:
            # Computational overhead
            f.write("COMPUTATIONAL OVERHEAD\n")
            f.write("-"*80 + "\n")
            
            # Extract timing information
            init_times = [timing.get('1_initialization', 0) for timing in success_df['timing']]
            eval_times = [timing.get('2_model_evaluation', 0) for timing in success_df['timing']]
            rank_times = [timing.get('3_model_ranking', 0) for timing in success_df['timing']]
            e2e_times = [timing.get('end_to_end', 0) for timing in success_df['timing']]
            
            f.write(f"Average Initialization Time: {np.mean(init_times):.2f}s (±{np.std(init_times):.2f}s)\n")
            f.write(f"Average Model Evaluation Time: {np.mean(eval_times):.2f}s (±{np.std(eval_times):.2f}s)\n")
            f.write(f"Average Model Ranking Time: {np.mean(rank_times):.2f}s (±{np.std(rank_times):.2f}s)\n")
            f.write(f"Average End-to-End Time: {np.mean(e2e_times):.2f}s (±{np.std(e2e_times):.2f}s)\n")
            f.write(f"Total Computational Time: {np.sum(e2e_times):.2f}s\n\n")
            
            # Memory usage
            f.write("MEMORY USAGE\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Peak Memory: {success_df['memory_peak_mb'].mean():.2f} MB\n")
            f.write(f"Max Peak Memory: {success_df['memory_peak_mb'].max():.2f} MB\n")
            f.write(f"Average Memory: {success_df['memory_avg_mb'].mean():.2f} MB\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            
            valid_f1 = success_df['best_f1_score'].dropna()
            valid_prauc = success_df['best_prauc_score'].dropna()
            
            if len(valid_f1) > 0:
                f.write(f"Average F1 Score: {valid_f1.mean():.4f} (±{valid_f1.std():.4f})\n")
                f.write(f"Median F1 Score: {valid_f1.median():.4f}\n")
            else:
                f.write("No valid F1 scores available\n")
            
            if len(valid_prauc) > 0:
                f.write(f"Average PR-AUC: {valid_prauc.mean():.4f} (±{valid_prauc.std():.4f})\n")
                f.write(f"Median PR-AUC: {valid_prauc.median():.4f}\n")
            else:
                f.write("No valid PR-AUC scores available\n")
            
            f.write("\n")
            
            # Per-domain statistics
            f.write("PER-DOMAIN STATISTICS\n")
            f.write("-"*80 + "\n")
            
            for domain in sorted(success_df['domain'].unique()):
                domain_df = success_df[success_df['domain'] == domain]
                f.write(f"\n{domain}:\n")
                f.write(f"  Datasets: {len(domain_df)}\n")
                
                domain_e2e = [timing.get('end_to_end', 0) for timing in domain_df['timing']]
                f.write(f"  Avg E2E Time: {np.mean(domain_e2e):.2f}s\n")
                f.write(f"  Avg Peak Memory: {domain_df['memory_peak_mb'].mean():.2f} MB\n")
                
                domain_f1 = domain_df['best_f1_score'].dropna()
                if len(domain_f1) > 0:
                    f.write(f"  Avg F1: {domain_f1.mean():.4f}\n")
                
                domain_prauc = domain_df['best_prauc_score'].dropna()
                if len(domain_prauc) > 0:
                    f.write(f"  Avg PR-AUC: {domain_prauc.mean():.4f}\n")
        else:
            f.write("No successful runs to summarize.\n")
        
        # Failed runs
        failed_df = results_df[~results_df['success']]
        if len(failed_df) > 0:
            f.write("\n" + "="*80 + "\n")
            f.write("FAILED RUNS\n")
            f.write("-"*80 + "\n")
            for _, row in failed_df.iterrows():
                f.write(f"{row['file_name']}: {row['error']}\n")
    
    print(f"Summary report saved to {summary_file}")
    
    # Also regenerate CSV
    csv_file = os.path.join(results_dir, 'results.csv')
    csv_data = []
    for result in results:
        csv_row = {
            'file_name': result['file_name'],
            'domain': result['domain'],
            'success': result['success'],
            'error': result['error'],
            'memory_peak_mb': result.get('memory_peak_mb'),
            'memory_avg_mb': result.get('memory_avg_mb'),
            'best_model_prauc': result.get('best_model_prauc'),
            'best_model_f1': result.get('best_model_f1'),
            'best_prauc_score': result.get('best_prauc_score'),
            'best_f1_score': result.get('best_f1_score'),
        }
        
        # Add timing columns if available
        if 'timing' in result and result['timing']:
            csv_row.update({
                'time_1_initialization': result['timing'].get('1_initialization'),
                'time_2_model_evaluation': result['timing'].get('2_model_evaluation'),
                'time_3_model_ranking': result['timing'].get('3_model_ranking'),
                'time_end_to_end': result['timing'].get('end_to_end'),
            })
        
        csv_data.append(csv_row)
    
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    print(f"CSV report saved to {csv_file}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to latest results
        results_dir = '/home/maxoud/local-storage/projects/ums-tsad/testbed_results/test_results_3datasets_20260104_141020'
    
    regenerate_summary(results_dir)
