#!/usr/bin/env python3
"""
Compare UMS-TSAD Baseline with RAMSeS Results

This script compares the computational overhead and performance metrics
between the UMS-TSAD baseline (3 criteria: centrality, synthetic anomalies,
forecasting) and the RAMSeS framework (ensemble GA + robust single model
selection with LinTS, GAN tests, borderline tests, Monte Carlo, etc.).

Author: Testbed evaluation comparison
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse

# Setup plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class BaselineRAMSeSComparator:
    """Compare UMS-TSAD baseline with RAMSeS results"""
    
    def __init__(self, baseline_dir: str, ramses_dir: str, output_dir: str = None):
        """
        Initialize comparator
        
        Parameters
        ----------
        baseline_dir : str
            Directory with UMS-TSAD baseline results
        ramses_dir : str
            Directory with RAMSeS results
        output_dir : str, optional
            Output directory for comparison plots and reports
        """
        self.baseline_dir = baseline_dir
        self.ramses_dir = ramses_dir
        
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"comparison_results_{timestamp}"
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading UMS-TSAD baseline results from: {baseline_dir}")
        print(f"Loading RAMSeS results from: {ramses_dir}")
        print(f"Output directory: {output_dir}")
    
    def load_baseline_results(self) -> pd.DataFrame:
        """Load UMS-TSAD baseline results"""
        csv_file = os.path.join(self.baseline_dir, 'results.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Baseline results not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        df['framework'] = 'UMS-TSAD (Baseline)'
        
        print(f"Loaded {len(df)} baseline results")
        return df
    
    def load_ramses_results(self) -> pd.DataFrame:
        """Load RAMSeS results"""
        # Try different possible RAMSeS result formats
        csv_file = os.path.join(self.ramses_dir, 'results.csv')
        
        if not os.path.exists(csv_file):
            # Try alternative location
            csv_file = os.path.join(self.ramses_dir, 'summary.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"RAMSeS results not found in: {self.ramses_dir}")
        
        df = pd.read_csv(csv_file)
        df['framework'] = 'RAMSeS'
        
        print(f"Loaded {len(df)} RAMSeS results")
        return df
    
    def compute_overhead_comparison(self, baseline_df: pd.DataFrame, ramses_df: pd.DataFrame) -> dict:
        """
        Compute computational overhead comparison
        
        Returns
        -------
        Dict with overhead statistics
        """
        comparison = {}
        
        # UMS-TSAD timing breakdown
        if 'time_end_to_end' in baseline_df.columns:
            baseline_e2e = baseline_df['time_end_to_end'].dropna()
        elif 'time_1_initialization' in baseline_df.columns:
            # Sum up components
            baseline_e2e = (
                baseline_df['time_1_initialization'].fillna(0) +
                baseline_df['time_2_model_evaluation'].fillna(0) +
                baseline_df['time_3_model_ranking'].fillna(0)
            )
        else:
            baseline_e2e = pd.Series([])
        
        # RAMSeS timing
        if 'time_end_to_end' in ramses_df.columns:
            ramses_e2e = ramses_df['time_end_to_end'].dropna()
        else:
            ramses_e2e = pd.Series([])
        
        if len(baseline_e2e) > 0:
            comparison['baseline'] = {
                'mean_e2e': baseline_e2e.mean(),
                'std_e2e': baseline_e2e.std(),
                'median_e2e': baseline_e2e.median(),
                'total_e2e': baseline_e2e.sum(),
                'count': len(baseline_e2e)
            }
        
        if len(ramses_e2e) > 0:
            comparison['ramses'] = {
                'mean_e2e': ramses_e2e.mean(),
                'std_e2e': ramses_e2e.std(),
                'median_e2e': ramses_e2e.median(),
                'total_e2e': ramses_e2e.sum(),
                'count': len(ramses_e2e)
            }
        
        # Compute relative overhead
        if 'baseline' in comparison and 'ramses' in comparison:
            baseline_mean = comparison['baseline']['mean_e2e']
            ramses_mean = comparison['ramses']['mean_e2e']
            
            comparison['relative'] = {
                'ramses_overhead_factor': ramses_mean / baseline_mean if baseline_mean > 0 else float('inf'),
                'ramses_overhead_seconds': ramses_mean - baseline_mean,
                'ramses_overhead_percent': ((ramses_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else float('inf')
            }
        
        return comparison
    
    def compute_performance_comparison(self, baseline_df: pd.DataFrame, ramses_df: pd.DataFrame) -> dict:
        """
        Compute performance metric comparison
        
        Returns
        -------
        Dict with performance statistics
        """
        comparison = {}
        
        # F1 scores
        baseline_f1 = baseline_df['best_f1_score'].dropna()
        
        # Try different column names for RAMSeS
        ramses_f1_cols = ['final_f1', 'best_f1_score', 'f1_score', 'final_decision_f1']
        ramses_f1 = pd.Series([])
        for col in ramses_f1_cols:
            if col in ramses_df.columns:
                ramses_f1 = ramses_df[col].dropna()
                break
        
        if len(baseline_f1) > 0:
            comparison['baseline_f1'] = {
                'mean': baseline_f1.mean(),
                'std': baseline_f1.std(),
                'median': baseline_f1.median(),
                'min': baseline_f1.min(),
                'max': baseline_f1.max()
            }
        
        if len(ramses_f1) > 0:
            comparison['ramses_f1'] = {
                'mean': ramses_f1.mean(),
                'std': ramses_f1.std(),
                'median': ramses_f1.median(),
                'min': ramses_f1.min(),
                'max': ramses_f1.max()
            }
        
        # PR-AUC scores
        baseline_prauc = baseline_df['best_prauc_score'].dropna()
        
        ramses_prauc_cols = ['final_prauc', 'best_prauc_score', 'pr_auc', 'final_decision_prauc']
        ramses_prauc = pd.Series([])
        for col in ramses_prauc_cols:
            if col in ramses_df.columns:
                ramses_prauc = ramses_df[col].dropna()
                break
        
        if len(baseline_prauc) > 0:
            comparison['baseline_prauc'] = {
                'mean': baseline_prauc.mean(),
                'std': baseline_prauc.std(),
                'median': baseline_prauc.median(),
                'min': baseline_prauc.min(),
                'max': baseline_prauc.max()
            }
        
        if len(ramses_prauc) > 0:
            comparison['ramses_prauc'] = {
                'mean': ramses_prauc.mean(),
                'std': ramses_prauc.std(),
                'median': ramses_prauc.median(),
                'min': ramses_prauc.min(),
                'max': ramses_prauc.max()
            }
        
        return comparison
    
    def plot_overhead_comparison(self, baseline_df: pd.DataFrame, ramses_df: pd.DataFrame):
        """Create overhead comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract timing data
        baseline_e2e = baseline_df.get('time_end_to_end', 
                                       baseline_df.get('time_1_initialization', pd.Series([])))
        ramses_e2e = ramses_df.get('time_end_to_end', pd.Series([]))
        
        # Plot 1: Box plot comparison
        data = []
        if len(baseline_e2e) > 0:
            data.extend([{'Framework': 'UMS-TSAD\n(Baseline)', 'Time (s)': t} for t in baseline_e2e])
        if len(ramses_e2e) > 0:
            data.extend([{'Framework': 'RAMSeS', 'Time (s)': t} for t in ramses_e2e])
        
        if data:
            plot_df = pd.DataFrame(data)
            sns.boxplot(x='Framework', y='Time (s)', data=plot_df, ax=axes[0])
            axes[0].set_title('End-to-End Time Comparison')
            axes[0].set_ylabel('Time (seconds)')
        
        # Plot 2: Memory comparison
        baseline_mem = baseline_df['memory_peak_mb'].dropna()
        ramses_mem = ramses_df.get('memory_peak_mb', ramses_df.get('peak_memory_mb', pd.Series([])))
        
        data = []
        if len(baseline_mem) > 0:
            data.extend([{'Framework': 'UMS-TSAD\n(Baseline)', 'Memory (MB)': m} for m in baseline_mem])
        if len(ramses_mem) > 0:
            data.extend([{'Framework': 'RAMSeS', 'Memory (MB)': m} for m in ramses_mem])
        
        if data:
            plot_df = pd.DataFrame(data)
            sns.boxplot(x='Framework', y='Memory (MB)', data=plot_df, ax=axes[1])
            axes[1].set_title('Peak Memory Usage Comparison')
            axes[1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'overhead_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved overhead comparison plot: {output_file}")
        plt.close()
    
    def plot_performance_comparison(self, baseline_df: pd.DataFrame, ramses_df: pd.DataFrame):
        """Create performance comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # F1 scores
        baseline_f1 = baseline_df['best_f1_score'].dropna()
        ramses_f1 = ramses_df.get('final_f1', ramses_df.get('best_f1_score', pd.Series([])))
        
        data = []
        if len(baseline_f1) > 0:
            data.extend([{'Framework': 'UMS-TSAD\n(Baseline)', 'F1 Score': f} for f in baseline_f1])
        if len(ramses_f1) > 0:
            data.extend([{'Framework': 'RAMSeS', 'F1 Score': f} for f in ramses_f1])
        
        if data:
            plot_df = pd.DataFrame(data)
            sns.boxplot(x='Framework', y='F1 Score', data=plot_df, ax=axes[0])
            axes[0].set_title('F1 Score Comparison')
            axes[0].set_ylabel('F1 Score')
        
        # PR-AUC
        baseline_prauc = baseline_df['best_prauc_score'].dropna()
        ramses_prauc = ramses_df.get('final_prauc', ramses_df.get('best_prauc_score', pd.Series([])))
        
        data = []
        if len(baseline_prauc) > 0:
            data.extend([{'Framework': 'UMS-TSAD\n(Baseline)', 'PR-AUC': p} for p in baseline_prauc])
        if len(ramses_prauc) > 0:
            data.extend([{'Framework': 'RAMSeS', 'PR-AUC': p} for p in ramses_prauc])
        
        if data:
            plot_df = pd.DataFrame(data)
            sns.boxplot(x='Framework', y='PR-AUC', data=plot_df, ax=axes[1])
            axes[1].set_title('PR-AUC Comparison')
            axes[1].set_ylabel('PR-AUC')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot: {output_file}")
        plt.close()
    
    def generate_report(self, overhead_comp: dict, performance_comp: dict):
        """Generate comprehensive comparison report"""
        report_file = os.path.join(self.output_dir, 'comparison_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UMS-TSAD BASELINE vs RAMSeS COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            
            if 'relative' in overhead_comp:
                overhead_factor = overhead_comp['relative']['ramses_overhead_factor']
                overhead_pct = overhead_comp['relative']['ramses_overhead_percent']
                
                f.write(f"Computational Overhead:\n")
                f.write(f"  RAMSeS is {overhead_factor:.2f}x slower than UMS-TSAD baseline\n")
                f.write(f"  RAMSeS overhead: {overhead_pct:.1f}%\n")
                f.write(f"  Additional time per dataset: {overhead_comp['relative']['ramses_overhead_seconds']:.2f}s\n")
                f.write("\n")
            
            if 'baseline_f1' in performance_comp and 'ramses_f1' in performance_comp:
                baseline_f1 = performance_comp['baseline_f1']['mean']
                ramses_f1 = performance_comp['ramses_f1']['mean']
                f1_improvement = ((ramses_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                
                f.write(f"Performance Improvement (F1):\n")
                f.write(f"  UMS-TSAD Baseline: {baseline_f1:.4f}\n")
                f.write(f"  RAMSeS: {ramses_f1:.4f}\n")
                f.write(f"  Improvement: {f1_improvement:+.2f}%\n")
                f.write("\n")
            
            if 'baseline_prauc' in performance_comp and 'ramses_prauc' in performance_comp:
                baseline_prauc = performance_comp['baseline_prauc']['mean']
                ramses_prauc = performance_comp['ramses_prauc']['mean']
                prauc_improvement = ((ramses_prauc - baseline_prauc) / baseline_prauc * 100) if baseline_prauc > 0 else 0
                
                f.write(f"Performance Improvement (PR-AUC):\n")
                f.write(f"  UMS-TSAD Baseline: {baseline_prauc:.4f}\n")
                f.write(f"  RAMSeS: {ramses_prauc:.4f}\n")
                f.write(f"  Improvement: {prauc_improvement:+.2f}%\n")
                f.write("\n")
            
            # Detailed Computational Overhead
            f.write("="*80 + "\n")
            f.write("DETAILED COMPUTATIONAL OVERHEAD\n")
            f.write("="*80 + "\n\n")
            
            if 'baseline' in overhead_comp:
                f.write("UMS-TSAD Baseline (3 Criteria):\n")
                f.write(f"  Mean E2E Time: {overhead_comp['baseline']['mean_e2e']:.2f}s (±{overhead_comp['baseline']['std_e2e']:.2f}s)\n")
                f.write(f"  Median E2E Time: {overhead_comp['baseline']['median_e2e']:.2f}s\n")
                f.write(f"  Total Time: {overhead_comp['baseline']['total_e2e']:.2f}s\n")
                f.write(f"  Datasets: {overhead_comp['baseline']['count']}\n")
                f.write("\n")
            
            if 'ramses' in overhead_comp:
                f.write("RAMSeS (Full Pipeline):\n")
                f.write(f"  Mean E2E Time: {overhead_comp['ramses']['mean_e2e']:.2f}s (±{overhead_comp['ramses']['std_e2e']:.2f}s)\n")
                f.write(f"  Median E2E Time: {overhead_comp['ramses']['median_e2e']:.2f}s\n")
                f.write(f"  Total Time: {overhead_comp['ramses']['total_e2e']:.2f}s\n")
                f.write(f"  Datasets: {overhead_comp['ramses']['count']}\n")
                f.write("\n")
            
            # Performance Metrics
            f.write("="*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")
            
            if 'baseline_f1' in performance_comp:
                f.write("F1 Score - UMS-TSAD Baseline:\n")
                for key, val in performance_comp['baseline_f1'].items():
                    f.write(f"  {key}: {val:.4f}\n")
                f.write("\n")
            
            if 'ramses_f1' in performance_comp:
                f.write("F1 Score - RAMSeS:\n")
                for key, val in performance_comp['ramses_f1'].items():
                    f.write(f"  {key}: {val:.4f}\n")
                f.write("\n")
            
            if 'baseline_prauc' in performance_comp:
                f.write("PR-AUC - UMS-TSAD Baseline:\n")
                for key, val in performance_comp['baseline_prauc'].items():
                    f.write(f"  {key}: {val:.4f}\n")
                f.write("\n")
            
            if 'ramses_prauc' in performance_comp:
                f.write("PR-AUC - RAMSeS:\n")
                for key, val in performance_comp['ramses_prauc'].items():
                    f.write(f"  {key}: {val:.4f}\n")
                f.write("\n")
            
            # Interpretation
            f.write("="*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*80 + "\n\n")
            
            f.write("UMS-TSAD Baseline uses 3 model selection criteria:\n")
            f.write("  1. Centrality-based ranking (neighbor distance in score space)\n")
            f.write("  2. Synthetic anomaly injection (multiple anomaly types)\n")
            f.write("  3. Forecasting performance metrics (MAE, MSE, SMAPE, etc.)\n")
            f.write("\n")
            
            f.write("RAMSeS extends this with:\n")
            f.write("  - Ensemble optimization via Genetic Algorithm\n")
            f.write("  - Linear Thompson Sampling for online selection\n")
            f.write("  - GAN-based robustness testing\n")
            f.write("  - Borderline/off-by-threshold sensitivity tests\n")
            f.write("  - Monte Carlo noise stress-tests\n")
            f.write("  - Markov-chain rank aggregation\n")
            f.write("\n")
            
            if 'relative' in overhead_comp:
                f.write(f"The additional computational cost of {overhead_comp['relative']['ramses_overhead_percent']:.1f}% ")
                
                if 'baseline_f1' in performance_comp and 'ramses_f1' in performance_comp:
                    baseline_f1 = performance_comp['baseline_f1']['mean']
                    ramses_f1 = performance_comp['ramses_f1']['mean']
                    f1_improvement = ((ramses_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                    
                    if f1_improvement > 0:
                        f.write(f"yields a {f1_improvement:.2f}% improvement in F1 score.\n")
                    else:
                        f.write(f"results in a {abs(f1_improvement):.2f}% decrease in F1 score.\n")
                else:
                    f.write("for the RAMSeS framework.\n")
        
        print(f"Saved comparison report: {report_file}")
        
        # Also save as JSON for programmatic access
        json_file = os.path.join(self.output_dir, 'comparison_data.json')
        comparison_data = {
            'overhead': overhead_comp,
            'performance': performance_comp,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(json_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        print(f"Saved comparison data: {json_file}")
    
    def run_comparison(self):
        """Run full comparison pipeline"""
        print("\n" + "="*80)
        print("Starting UMS-TSAD vs RAMSeS Comparison")
        print("="*80 + "\n")
        
        # Load results
        baseline_df = self.load_baseline_results()
        ramses_df = self.load_ramses_results()
        
        # Compute comparisons
        print("\nComputing overhead comparison...")
        overhead_comp = self.compute_overhead_comparison(baseline_df, ramses_df)
        
        print("Computing performance comparison...")
        performance_comp = self.compute_performance_comparison(baseline_df, ramses_df)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_overhead_comparison(baseline_df, ramses_df)
        self.plot_performance_comparison(baseline_df, ramses_df)
        
        # Generate report
        print("\nGenerating comparison report...")
        self.generate_report(overhead_comp, performance_comp)
        
        print("\n" + "="*80)
        print("Comparison complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare UMS-TSAD baseline with RAMSeS results"
    )
    parser.add_argument(
        '--baseline_dir',
        type=str,
        required=True,
        help='Directory with UMS-TSAD baseline results'
    )
    parser.add_argument(
        '--ramses_dir',
        type=str,
        required=True,
        help='Directory with RAMSeS results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    comparator = BaselineRAMSeSComparator(
        baseline_dir=args.baseline_dir,
        ramses_dir=args.ramses_dir,
        output_dir=args.output_dir
    )
    
    comparator.run_comparison()


if __name__ == '__main__':
    main()
