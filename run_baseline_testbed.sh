#!/bin/bash
################################################################################
# UMS-TSAD Baseline Testbed Runner
#
# This script runs the UMS-TSAD framework over the RAMSeS testbed to measure
# end-to-end computational overhead as a baseline comparison.
#
# Usage:
#   ./run_baseline_testbed.sh [DATASET_LIST] [OPTIONS]
#
# Examples:
#   # Run on UCR Anomaly Archive (small sample)
#   ./run_baseline_testbed.sh ucr_sample
#
#   # Run on full UCR Anomaly Archive
#   ./run_baseline_testbed.sh ucr_full
#
#   # Run on SMD dataset
#   ./run_baseline_testbed.sh smd
#
#   # Run on SKAB dataset
#   ./run_baseline_testbed.sh skab
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths - CUSTOMIZE THESE FOR YOUR SETUP
RAMSES_DIR="/home/maxoud/local-storage/projects/RAMSeS"
UMS_TSAD_DIR="/home/maxoud/local-storage/projects/ums-tsad"
DATASET_BASE_PATH="/home/maxoud/local-storage/projects/RAMSeS/Mononito/datasets"
TRAINED_MODELS_BASE="/home/maxoud/local-storage/projects/RAMSeS/Mononito/trained_models"

# Parse arguments
DATASET_TYPE=${1:-"ucr_sample"}
MAX_DATASETS=${2:-""}

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}UMS-TSAD Baseline Testbed Runner${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Select dataset list file based on type
case $DATASET_TYPE in
    ucr_sample)
        DATASET_LIST="$RAMSES_DIR/testbed/file_list/ucr_sample_10.csv"
        OUTPUT_DIR="$UMS_TSAD_DIR/testbed_results/ucr_sample_$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}Running on: UCR Anomaly Archive (Sample - 10 datasets)${NC}"
        ;;
    ucr_full)
        DATASET_LIST="$RAMSES_DIR/testbed/file_list/test_u_ucr_anomaly_archive.csv"
        OUTPUT_DIR="$UMS_TSAD_DIR/testbed_results/ucr_full_$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}Running on: UCR Anomaly Archive (Full - 250 datasets)${NC}"
        ;;
    smd|SMD)
        DATASET_LIST="$RAMSES_DIR/testbed/file_list/test_m_smd.csv"
        OUTPUT_DIR="$UMS_TSAD_DIR/testbed_results/smd_$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}Running on: Server Machine Dataset (SMD)${NC}"
        ;;
    skab|SKAB)
        DATASET_LIST="$RAMSES_DIR/testbed/file_list/test_m_skab.csv"
        OUTPUT_DIR="$UMS_TSAD_DIR/testbed_results/skab_$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}Running on: SKAB Dataset${NC}"
        ;;
    *)
        # Assume it's a direct path to a CSV file
        DATASET_LIST="$DATASET_TYPE"
        OUTPUT_DIR="$UMS_TSAD_DIR/testbed_results/custom_$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}Running on: Custom dataset list${NC}"
        ;;
esac

echo -e "Dataset List: ${YELLOW}$DATASET_LIST${NC}"
echo -e "Output Directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Check if dataset list exists
if [ ! -f "$DATASET_LIST" ]; then
    echo -e "${RED}Error: Dataset list file not found: $DATASET_LIST${NC}"
    echo ""
    echo "Available dataset lists in RAMSeS testbed:"
    ls -1 "$RAMSES_DIR/testbed/file_list/"*.csv 2>/dev/null || echo "No CSV files found"
    exit 1
fi

# Check if trained models exist
if [ ! -d "$TRAINED_MODELS_BASE" ]; then
    echo -e "${YELLOW}Warning: Trained models directory not found: $TRAINED_MODELS_BASE${NC}"
    echo -e "${YELLOW}You may need to train models first or update the path.${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/testbed_run.log"

echo -e "${BLUE}Starting testbed run...${NC}"
echo -e "Logs will be saved to: ${YELLOW}$LOG_FILE${NC}"
echo ""

# Build command
CMD="python3 $UMS_TSAD_DIR/run_testbed_baseline.py \
    --dataset_list '$DATASET_LIST' \
    --trained_model_path '$TRAINED_MODELS_BASE' \
    --dataset_path '$DATASET_BASE_PATH' \
    --output_dir '$OUTPUT_DIR' \
    --downsampling 10 \
    --min_length 256 \
    --timeout 3600"

# Add max_datasets if specified
if [ ! -z "$MAX_DATASETS" ]; then
    CMD="$CMD --max_datasets $MAX_DATASETS"
    echo -e "${YELLOW}Limiting to $MAX_DATASETS datasets${NC}"
    echo ""
fi

# Run the testbed
echo -e "${GREEN}Executing:${NC}"
echo "$CMD"
echo ""

# Execute and capture both stdout and stderr
eval "$CMD" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Testbed run completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "Results saved to: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    echo "Files generated:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    
    # Display summary if available
    SUMMARY_FILE="$OUTPUT_DIR/summary_report.txt"
    if [ -f "$SUMMARY_FILE" ]; then
        echo -e "${BLUE}======================================${NC}"
        echo -e "${BLUE}Quick Summary:${NC}"
        echo -e "${BLUE}======================================${NC}"
        head -n 30 "$SUMMARY_FILE"
        echo ""
        echo -e "Full summary: ${YELLOW}$SUMMARY_FILE${NC}"
    fi
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Testbed run failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo -e "Check logs: ${YELLOW}$LOG_FILE${NC}"
    exit $EXIT_CODE
fi

echo ""
echo -e "${BLUE}To compare with RAMSeS results, run:${NC}"
echo -e "${YELLOW}python3 compare_baseline_ramses.py --baseline_dir '$OUTPUT_DIR' --ramses_dir '<ramses_results_dir>'${NC}"
echo ""
