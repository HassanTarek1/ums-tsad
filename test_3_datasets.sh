#!/bin/bash
################################################################################
# Quick Test: Run UMS-TSAD Baseline on 3 Datasets
#
# This script tests the baseline framework on just 3 UCR datasets
# to verify everything works before running the full testbed.
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}UMS-TSAD Baseline - Quick Test (3 datasets)${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Paths - CUSTOMIZE THESE FOR YOUR SETUP
UMS_TSAD_DIR="/home/maxoud/local-storage/projects/ums-tsad"
DATASET_LIST="$UMS_TSAD_DIR/test_3_datasets.csv"
DATASET_PATH="/home/maxoud/local-storage/projects/RAMSeS/Mononito/datasets"
TRAINED_MODELS_PATH="/home/maxoud/local-storage/projects/RAMSeS/Mononito/trained_models"
OUTPUT_DIR="$UMS_TSAD_DIR/testbed_results/test_results_3datasets_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Dataset List: ${YELLOW}$DATASET_LIST${NC}"
echo -e "  Datasets Path: ${YELLOW}$DATASET_PATH${NC}"
echo -e "  Trained Models: ${YELLOW}$TRAINED_MODELS_PATH${NC}"
echo -e "  Output: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Check if dataset list exists
if [ ! -f "$DATASET_LIST" ]; then
    echo -e "${RED}Error: Dataset list not found: $DATASET_LIST${NC}"
    exit 1
fi

# Check dataset list content
echo -e "${BLUE}Datasets to test:${NC}"
cat "$DATASET_LIST"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the test
echo -e "${BLUE}Starting test run...${NC}"
echo ""

cd "$UMS_TSAD_DIR"

python3 run_testbed_baseline.py \
    --dataset_list "$DATASET_LIST" \
    --trained_model_path "$TRAINED_MODELS_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --downsampling 10 \
    --min_length 256 \
    --timeout 600

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Test completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "Results saved to: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
    
    # Show results
    if [ -f "$OUTPUT_DIR/summary_report.txt" ]; then
        echo -e "${BLUE}Quick Summary:${NC}"
        head -n 40 "$OUTPUT_DIR/summary_report.txt"
        echo ""
        echo -e "Full report: ${YELLOW}$OUTPUT_DIR/summary_report.txt${NC}"
    fi
    
    if [ -f "$OUTPUT_DIR/results.csv" ]; then
        echo ""
        echo -e "${BLUE}Results CSV:${NC}"
        column -t -s, "$OUTPUT_DIR/results.csv" | head -n 10
    fi
else
    echo -e "${RED}Test failed with exit code $EXIT_CODE${NC}"
    echo -e "Check logs in: ${YELLOW}$OUTPUT_DIR${NC}"
    exit $EXIT_CODE
fi

echo ""
echo -e "${GREEN}✓ Test complete!${NC}"
