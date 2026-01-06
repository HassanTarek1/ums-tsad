#!/usr/bin/env python3
"""
Testbed Runner for UMS-TSAD Framework (Baseline for RAMSeS)

This script runs the UMS-TSAD framework (the original paper implementation)
over the same testbed as RAMSeS to measure end-to-end computational overhead
and performance as a baseline comparison.

The UMS-TSAD framework uses 3 main model selection criteria:
1. Centrality-based ranking
2. Synthetic anomaly injection
3. Forecasting metrics

Author: Adapted for testbed evaluation
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import torch as t
from pathlib import Path
import json
from datetime import datetime
import logging
import traceback
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model_selection.model_selection import RankModels
from datasets.load import load_data
from utils.model_selection_utils import rank_models
from model_trainer.train import TrainModels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.measurements = []
    
    def update(self):
        """Record current memory usage"""
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.measurements.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb
    
    def get_average(self):
        """Get average memory usage"""
        return np.mean(self.measurements) if self.measurements else 0
    
    def get_peak(self):
        """Get peak memory usage"""
        return self.peak_memory
    
    def reset(self):
        """Reset measurements"""
        self.measurements = []
        self.peak_memory = 0


class UMSTSADTestbedRunner:
    """Run UMS-TSAD baseline over RAMSeS testbed"""
    
    def __init__(self, 
                 dataset_list_file: str,
                 trained_model_path: str,
                 dataset_path: str,
                 output_base_dir: str = "testbed_results",
                 timeout: int = 3600,
                 downsampling: int = 10,
                 min_length: int = 256,
                 normalize: bool = True,
                 train_models: bool = False,
                 algorithm_list: List[str] = None):
        """
        Initialize UMS-TSAD testbed runner
        
        Parameters
        ----------
        dataset_list_file : str
            Path to CSV file with columns: file_name, domain_name
        trained_model_path : str
            Path to trained models directory
        dataset_path : str
            Path to datasets root directory
        output_base_dir : str
            Base directory for storing results
        timeout : int
            Timeout in seconds for each dataset execution
        downsampling : int
            Downsampling factor for data loading
        min_length : int
            Minimum sequence length
        normalize : bool
            Whether to normalize data
        train_models : bool
            Whether to train models automatically if not found
        algorithm_list : List[str], optional
            List of algorithms to train (default: standard UMS-TSAD set)
        """
        self.dataset_list_file = dataset_list_file
        self.trained_model_path = trained_model_path
        self.dataset_path = dataset_path
        self.output_base_dir = output_base_dir
        self.timeout = timeout
        self.downsampling = downsampling
        self.min_length = min_length
        self.normalize = normalize
        self.train_models_flag = train_models
        # Default algorithms: Use stable and reasonably fast models
        # NN - Neural Network (fast, stable)
        # LOF - Local Outlier Factor (fast, stable)  
        # MD - Mean Deviation (fast, stable)
        # KDE, ABOD, CBLOF, COF, SOS - PyOD-based models (stable)
        # 
        # Excluded by default due to issues:
        # - RNN: Has tensor dimension bugs with certain window sizes
        # - LSTMVAE: Very slow training (5-10 min per entity)
        # - DGHL: Very slow training (5-10 min per entity)
        #
        # Users can still train all models with: --algorithms DGHL RNN LSTMVAE NN LOF MD KDE ABOD CBLOF COF SOS
        self.algorithm_list = algorithm_list or ['NN', 'ABOD', 'CBLOF', 'COF']
        
        self.memory_monitor = MemoryMonitor()
        self.results_by_domain = defaultdict(list)
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Load dataset list
        self.datasets_df = pd.read_csv(dataset_list_file)
        logger.info(f"Loaded {len(self.datasets_df)} datasets from {dataset_list_file}")
        
        # Group by domain
        self.domains = self.datasets_df['domain_name'].unique()
        logger.info(f"Found {len(self.domains)} domains: {list(self.domains)}")
        
        # Model configuration
        self.model_name_list = ['LSTMVAE', 'DGHL', 'NN', 'RNN', 'LOF', 'MD', 'CBLOF']
        # Use valid anomaly types from ANOMALY_PARAM_GRID: spikes, contextual, flip, speedup, noise, cutoff, scale, wander, average
        # Select 4 diverse types that cover different anomaly characteristics
        self.inject_abn_list = ['spikes', 'contextual', 'scale', 'wander']
        
    def convert_entity_name(self, file_name: str, domain: str) -> Tuple[str, str]:
        """
        Convert RAMSeS file naming to UMS-TSAD dataset/entity format
        
        Parameters
        ----------
        file_name : str
            Filename from testbed list (e.g., "001_UCR_Anomaly_XXX.txt")
        domain : str
            Domain name (e.g., "anomaly_archive", "SMD", "SKAB")
            
        Returns
        -------
        dataset_name : str
            Dataset identifier for load_data
        entity_name : str
            Entity identifier for load_data
        """
        # Remove file extension
        entity = file_name.replace('.txt', '').replace('.csv', '')
        
        # Normalize domain to lowercase for consistency with RAMSeS
        # This fixes the SKAB case sensitivity issue
        domain_lower = domain.lower()
        
        # Determine dataset format
        if domain_lower == 'anomaly_archive':
            dataset_name = 'anomaly_archive'
            # Entity is the full filename without extension
            entity_name = entity
        elif domain_lower == 'smd':
            dataset_name = 'smd'
            # SMD format: machine-X-Y -> use as-is
            entity_name = entity
        elif domain_lower == 'skab':
            dataset_name = 'skab'
            entity_name = entity
        else:
            # Default fallback - use lowercase
            dataset_name = domain_lower
            entity_name = entity
            
        return dataset_name, entity_name
    
    def train_models_if_needed(self, dataset_name: str, entity_name: str) -> bool:
        """
        Train models for a dataset/entity if they don't exist or if training is forced
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset (lowercase, e.g., 'skab')
        entity_name : str
            Name of the entity
            
        Returns
        -------
        bool
            True if models exist or were successfully trained, False otherwise
        """
        # Check for models in a case-insensitive manner under trained_model_path
        models_exist = False
        models_dir = None
        try:
            for d in os.listdir(self.trained_model_path):
                if d.lower() == dataset_name.lower():
                    candidate = os.path.join(self.trained_model_path, d, entity_name)
                    # Ensure the dataset directory contains files (models)
                    if os.path.exists(candidate) and os.listdir(os.path.join(self.trained_model_path, d)):
                        models_exist = True
                        models_dir = candidate
                        break
        except Exception:
            # If trained_model_path doesn't exist or isn't listable, fall back to original checks
            pass

        # Fallback: check explicit lowercase and UPPERCASE paths as before
        if not models_exist:
            models_dir_lower = os.path.join(self.trained_model_path, dataset_name, entity_name)
            models_dir_upper = os.path.join(self.trained_model_path, dataset_name.upper(), entity_name)
            if os.path.exists(models_dir_lower) and os.listdir(models_dir_lower):
                models_exist = True
                models_dir = models_dir_lower
            elif os.path.exists(models_dir_upper) and os.listdir(models_dir_upper):
                models_exist = True
                models_dir = models_dir_upper
        
        if models_exist and not self.train_models_flag:
            logger.info(f"Models found for {dataset_name}/{entity_name} at {models_dir}, skipping training")
            return True
        
        if not self.train_models_flag:
            logger.warning(f"No models found for {dataset_name}/{entity_name} and training is disabled")
            return False
        
        # Map lowercase dataset names to proper case for load_data
        dataset_name_for_load = dataset_name.upper() if dataset_name.lower() == 'skab' else dataset_name
        
        # Train models
        logger.info(f"Training models for {dataset_name}/{entity_name}...")
        try:
            trainer = TrainModels(
                dataset=dataset_name_for_load,  # Use proper case for load_data
                entity=entity_name,
                algorithm_list=self.algorithm_list,
                downsampling=self.downsampling,
                min_length=self.min_length,
                root_dir=self.dataset_path,
                training_size=1.0,
                overwrite=self.train_models_flag,  # Overwrite if training is forced
                verbose=True,
                save_dir=self.trained_model_path
            )
            
            # Monkey-patch to save to lowercase directory
            trainer.logging_hierarchy = [dataset_name, entity_name]
            
            trainer.train_models(model_architectures=self.algorithm_list)
            logger.info(f"✓ Successfully trained models for {dataset_name}/{entity_name}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to train models for {dataset_name}/{entity_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_single_dataset(self, file_name: str, domain: str) -> Dict:
        """
        Run UMS-TSAD on a single dataset
        
        Parameters
        ----------
        file_name : str
            Dataset filename
        domain : str
            Domain name
            
        Returns
        -------
        Dict with results and timing
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {file_name} (Domain: {domain})")
        logger.info(f"{'='*80}\n")
        
        # Convert naming
        dataset_name, entity_name = self.convert_entity_name(file_name, domain)
        
        # Initialize timing
        timing_dict = {}
        total_start = time.time()
        
        # Reset memory monitor
        self.memory_monitor.reset()
        self.memory_monitor.update()
        
        try:
            # ============================================================
            # STAGE 0: Train models if needed
            # ============================================================
            if self.train_models_flag:
                stage_start = time.time()
                logger.info("Stage 0: Training models...")
                models_ready = self.train_models_if_needed(dataset_name, entity_name)
                if not models_ready:
                    raise FileNotFoundError(f"Models not available for {dataset_name}/{entity_name}")
                timing_dict['0_training'] = time.time() - stage_start
                logger.info(f"Training complete: {timing_dict['0_training']:.2f}s")
                self.memory_monitor.update()
            else:
                # Check for models in a case-insensitive manner under trained_model_path
                models_exist = False
                models_dir_checked = []
                try:
                    for d in os.listdir(self.trained_model_path):
                        if d.lower() == dataset_name.lower():
                            candidate = os.path.join(self.trained_model_path, d, entity_name)
                            models_dir_checked.append(candidate)
                            if os.path.exists(candidate):
                                models_exist = True
                                break
                except Exception:
                    pass

                # Fallback: explicit lower/upper checks
                models_dir_lower = os.path.join(self.trained_model_path, dataset_name, entity_name)
                models_dir_upper = os.path.join(self.trained_model_path, dataset_name.upper(), entity_name)
                models_dir_checked.extend([models_dir_lower, models_dir_upper])
                if not models_exist:
                    if os.path.exists(models_dir_lower) or os.path.exists(models_dir_upper):
                        models_exist = True

                if not models_exist:
                    raise FileNotFoundError(f"Models not found for {dataset_name}/{entity_name} (checked {', '.join(models_dir_checked + [models_dir_lower, models_dir_upper])})")
            
            # ============================================================
            # STAGE 1: Initialize RankModels
            # ============================================================
            stage_start = time.time()
            logger.info("Stage 1: Initializing RankModels...")
            
            ranker = RankModels(
                dataset=dataset_name,
                entity=entity_name,
                model_name_list=self.model_name_list,
                inject_abn_list=self.inject_abn_list,
                trained_model_path=self.trained_model_path,
                downsampling=self.downsampling,
                min_length=self.min_length,
                root_dir=self.dataset_path,
                normalize=self.normalize,
                verbose=False
            )
            
            timing_dict['1_initialization'] = time.time() - stage_start
            self.memory_monitor.update()
            logger.info(f"Initialization complete: {timing_dict['1_initialization']:.2f}s")
            
            # ============================================================
            # STAGE 2: Evaluate Models
            # ============================================================
            stage_start = time.time()
            logger.info("Stage 2: Evaluating models with 3 criteria...")
            
            # Evaluate with centrality, synthetic anomalies, and forecasting
            models_performance_matrix = ranker.evaluate_models(
                n_neighbors=[2, 4, 6],  # Centrality neighbors
                n_repeats=3,            # Synthetic anomaly repeats
                split='test',
                synthetic_ranking_criterion='prauc',
                n_splits=100
            )
            
            timing_dict['2_model_evaluation'] = time.time() - stage_start
            self.memory_monitor.update()
            logger.info(f"Model evaluation complete: {timing_dict['2_model_evaluation']:.2f}s")
            
            # ============================================================
            # STAGE 3: Rank Models
            # ============================================================
            stage_start = time.time()
            logger.info("Stage 3: Ranking models...")
            
            ranks_by_metrics, rank_prauc, rank_f1, rank_vus = ranker.rank_models()
            
            timing_dict['3_model_ranking'] = time.time() - stage_start
            self.memory_monitor.update()
            logger.info(f"Model ranking complete: {timing_dict['3_model_ranking']:.2f}s")
            
            # ============================================================
            # Compute Overall Timing
            # ============================================================
            total_time = time.time() - total_start
            timing_dict['end_to_end'] = total_time
            
            # Get best model by PR-AUC ranking
            # rank_prauc/rank_f1 are numpy arrays where each element is the rank (0=best)
            # We need to find which model index has rank 0 (the best rank)
            if len(rank_prauc) > 0:
                best_idx_prauc = int(np.argmin(rank_prauc))
                best_model_prauc = ranker.models_performance_matrix.index[best_idx_prauc]
            else:
                best_model_prauc = 'Unknown'
                
            if len(rank_f1) > 0:
                best_idx_f1 = int(np.argmin(rank_f1))
                best_model_f1 = ranker.models_performance_matrix.index[best_idx_f1]
            else:
                best_model_f1 = 'Unknown'
            
            # Extract performance metrics for best models
            # ranker.models_prauc and ranker.models_f1 are DataFrames with model names as index
            if best_model_prauc != 'Unknown' and best_model_prauc in ranker.models_prauc.index:
                best_prauc_score = ranker.models_prauc.loc[best_model_prauc, 'PR-AUC']
            else:
                best_prauc_score = -1
                
            if best_model_f1 != 'Unknown' and best_model_f1 in ranker.models_f1.index:
                best_f1_score = ranker.models_f1.loc[best_model_f1, 'Best F-1']
            else:
                best_f1_score = -1
            
            logger.info(f"\nBest Model (PR-AUC): {best_model_prauc} (Score: {best_prauc_score:.4f})")
            logger.info(f"Best Model (F1): {best_model_f1} (Score: {best_f1_score:.4f})")
            logger.info(f"\nEnd-to-End Time: {total_time:.2f}s")
            logger.info(f"Peak Memory: {self.memory_monitor.get_peak():.2f} MB")
            
            # Compile results
            results = {
                'file_name': file_name,
                'domain': domain,
                'dataset_name': dataset_name,
                'entity_name': entity_name,
                'success': True,
                'error': None,
                'timing': timing_dict,
                'memory_peak_mb': self.memory_monitor.get_peak(),
                'memory_avg_mb': self.memory_monitor.get_average(),
                'best_model_prauc': best_model_prauc,
                'best_model_f1': best_model_f1,
                'best_prauc_score': float(best_prauc_score) if best_prauc_score != -1 else None,
                'best_f1_score': float(best_f1_score) if best_f1_score != -1 else None,
                'models_evaluated': len(ranker.MODEL_NAMES),
                'performance_matrix': models_performance_matrix.to_dict(),
                'rank_prauc': rank_prauc.to_list() if hasattr(rank_prauc, 'to_list') else list(rank_prauc),
                'rank_f1': rank_f1.to_list() if hasattr(rank_f1, 'to_list') else list(rank_f1)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'file_name': file_name,
                'domain': domain,
                'dataset_name': dataset_name,
                'entity_name': entity_name,
                'success': False,
                'error': str(e),
                'timing': timing_dict,
                'memory_peak_mb': self.memory_monitor.get_peak(),
                'memory_avg_mb': self.memory_monitor.get_average()
            }
    
    def run_testbed(self, max_datasets: int = None) -> pd.DataFrame:
        """
        Run testbed over all datasets
        
        Parameters
        ----------
        max_datasets : int, optional
            Maximum number of datasets to process (for testing)
            
        Returns
        -------
        DataFrame with all results
        """
        all_results = []
        
        # Process datasets
        datasets_to_process = self.datasets_df.head(max_datasets) if max_datasets else self.datasets_df
        
        for idx, row in datasets_to_process.iterrows():
            file_name = row['file_name']
            domain = row['domain_name']
            
            # Run single dataset
            result = self.run_single_dataset(file_name, domain)
            all_results.append(result)
            
            # Store by domain
            self.results_by_domain[domain].append(result)
            
            # Save intermediate results
            self.save_results(all_results)
            
            logger.info(f"\nProgress: {idx + 1}/{len(datasets_to_process)} datasets processed")
        
        # Create summary
        results_df = pd.DataFrame(all_results)
        self.save_summary(results_df)
        
        return results_df
    
    def save_results(self, results: List[Dict]):
        """Save detailed results to JSON, organized by domain"""
        # Group results by domain
        results_by_domain = defaultdict(list)
        for result in results:
            domain = result.get('domain', 'unknown').lower()
            results_by_domain[domain].append(result)
        
        # Save results for each domain
        for domain, domain_results in results_by_domain.items():
            domain_dir = os.path.join(self.output_base_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            output_file = os.path.join(domain_dir, 'detailed_results.json')
            with open(output_file, 'w') as f:
                json.dump(domain_results, f, indent=2, default=str)
            
            logger.info(f"Saved detailed results to {output_file}")
    
    def save_summary(self, results_df: pd.DataFrame):
        """Generate and save summary statistics, organized by domain"""
        
        # Save overall summary
        summary_file = os.path.join(self.output_base_dir, 'overall_summary.txt')
        
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
                
                if len(valid_prauc) > 0:
                    f.write(f"Average PR-AUC: {valid_prauc.mean():.4f} (±{valid_prauc.std():.4f})\n")
                    f.write(f"Median PR-AUC: {valid_prauc.median():.4f}\n")
                
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
        
        logger.info(f"Saved overall summary to {summary_file}")
        
        # Save per-domain summaries and CSVs
        for domain in results_df['domain'].unique():
            domain_lower = domain.lower()
            domain_dir = os.path.join(self.output_base_dir, domain_lower)
            os.makedirs(domain_dir, exist_ok=True)
            
            domain_df = results_df[results_df['domain'] == domain]
            
            # Domain-specific summary
            domain_summary_file = os.path.join(domain_dir, 'summary_report.txt')
            with open(domain_summary_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"UMS-TSAD Results Summary - {domain.upper()}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Total Datasets: {len(domain_df)}\n")
                f.write(f"Successful: {domain_df['success'].sum()}\n")
                f.write(f"Failed: {(~domain_df['success']).sum()}\n\n")
                
                success_domain_df = domain_df[domain_df['success']]
                
                if len(success_domain_df) > 0:
                    # Timing
                    init_times = [timing.get('1_initialization', 0) for timing in success_domain_df['timing']]
                    eval_times = [timing.get('2_model_evaluation', 0) for timing in success_domain_df['timing']]
                    rank_times = [timing.get('3_model_ranking', 0) for timing in success_domain_df['timing']]
                    e2e_times = [timing.get('end_to_end', 0) for timing in success_domain_df['timing']]
                    
                    f.write("COMPUTATIONAL OVERHEAD\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Average Initialization Time: {np.mean(init_times):.2f}s (±{np.std(init_times):.2f}s)\n")
                    f.write(f"Average Model Evaluation Time: {np.mean(eval_times):.2f}s (±{np.std(eval_times):.2f}s)\n")
                    f.write(f"Average Model Ranking Time: {np.mean(rank_times):.2f}s (±{np.std(rank_times):.2f}s)\n")
                    f.write(f"Average End-to-End Time: {np.mean(e2e_times):.2f}s (±{np.std(e2e_times):.2f}s)\n")
                    f.write(f"Total Time: {np.sum(e2e_times):.2f}s\n\n")
                    
                    # Memory
                    f.write("MEMORY USAGE\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Average Peak Memory: {success_domain_df['memory_peak_mb'].mean():.2f} MB\n")
                    f.write(f"Max Peak Memory: {success_domain_df['memory_peak_mb'].max():.2f} MB\n\n")
                    
                    # Performance
                    valid_f1 = success_domain_df['best_f1_score'].dropna()
                    valid_prauc = success_domain_df['best_prauc_score'].dropna()
                    
                    if len(valid_f1) > 0 or len(valid_prauc) > 0:
                        f.write("PERFORMANCE METRICS\n")
                        f.write("-"*80 + "\n")
                        if len(valid_f1) > 0:
                            f.write(f"Average F1 Score: {valid_f1.mean():.4f} (±{valid_f1.std():.4f})\n")
                            f.write(f"Median F1 Score: {valid_f1.median():.4f}\n")
                        if len(valid_prauc) > 0:
                            f.write(f"Average PR-AUC: {valid_prauc.mean():.4f} (±{valid_prauc.std():.4f})\n")
                            f.write(f"Median PR-AUC: {valid_prauc.median():.4f}\n")
            
            logger.info(f"Saved {domain} summary to {domain_summary_file}")
            
            # Domain-specific CSV
            csv_file = os.path.join(domain_dir, 'results.csv')
            csv_data = []
            for _, row in domain_df.iterrows():
                csv_row = {
                    'file_name': row['file_name'],
                    'domain': row['domain'],
                    'success': row['success'],
                    'error': row['error'],
                    'memory_peak_mb': row['memory_peak_mb'],
                    'memory_avg_mb': row['memory_avg_mb'],
                    'best_model_prauc': row.get('best_model_prauc'),
                    'best_model_f1': row.get('best_model_f1'),
                    'best_prauc_score': row.get('best_prauc_score'),
                    'best_f1_score': row.get('best_f1_score'),
                }
                
                # Add timing columns
                if 'timing' in row and isinstance(row['timing'], dict):
                    for key, val in row['timing'].items():
                        csv_row[f'time_{key}'] = val
                
                csv_data.append(csv_row)
            
            csv_df = pd.DataFrame(csv_data)
            csv_df.to_csv(csv_file, index=False)
            logger.info(f"Saved {domain} CSV to {csv_file}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run UMS-TSAD baseline over RAMSeS testbed"
    )
    parser.add_argument(
        '--dataset_list',
        type=str,
        required=True,
        help='Path to dataset list CSV file (e.g., testbed/file_list/test_u_ucr_anomaly_archive.csv)'
    )
    parser.add_argument(
        '--trained_model_path',
        type=str,
        required=True,
        help='Path to trained models directory'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to datasets root directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='testbed_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max_datasets',
        type=int,
        default=None,
        help='Maximum number of datasets to process (for testing)'
    )
    parser.add_argument(
        '--downsampling',
        type=int,
        default=10,
        help='Downsampling factor'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=256,
        help='Minimum sequence length'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=36000,
        help='Timeout per dataset in seconds'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train models automatically if not found or retrain all models'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        default=None,  # Will use default list from __init__
        help='List of algorithms to train (e.g., --algorithms NN LOF DGHL). Default: NN LOF MD KDE ABOD CBLOF COF SOS'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = UMSTSADTestbedRunner(
        dataset_list_file=args.dataset_list,
        trained_model_path=args.trained_model_path,
        dataset_path=args.dataset_path,
        output_base_dir=args.output_dir,
        timeout=args.timeout,
        downsampling=args.downsampling,
        min_length=args.min_length,
        train_models=args.train,
        algorithm_list=args.algorithms
    )
    
    # Run testbed
    logger.info("Starting UMS-TSAD baseline testbed...")
    results_df = runner.run_testbed(max_datasets=args.max_datasets)
    
    logger.info(f"\nTestbed complete! Results saved to {args.output_dir}")
    logger.info(f"Processed {len(results_df)} datasets")
    logger.info(f"Success rate: {results_df['success'].mean()*100:.1f}%")


if __name__ == '__main__':
    main()
