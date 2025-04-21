#!/bin/bash

# This script splits the original Charades-Ego training annotation file
# into a new training set (90%) and a test set (10%).

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
ANNOTATION_DIR="data/CharadesEgo" # Directory containing the annotation files
ORIGINAL_TRAIN_FILE="charadesego_train_revised_v2_grouped.txt"
NEW_TRAIN_FILE="charadesego_train_90.txt"
TEST_FILE="charadesego_test_10.txt"
SPLIT_RATIO=0.1 # 10% for the test set

# --- Paths ---
ORIGINAL_TRAIN_PATH="${ANNOTATION_DIR}/${ORIGINAL_TRAIN_FILE}"
NEW_TRAIN_PATH="${ANNOTATION_DIR}/${NEW_TRAIN_FILE}"
TEST_PATH="${ANNOTATION_DIR}/${TEST_FILE}"

# --- Check if input file exists ---
if [ ! -f "${ORIGINAL_TRAIN_PATH}" ]; then
    echo "Error: Original training file not found at ${ORIGINAL_TRAIN_PATH}"
    exit 1
fi

# --- Calculate split sizes ---
TOTAL_LINES=$(wc -l < "${ORIGINAL_TRAIN_PATH}")
if [ "${TOTAL_LINES}" -eq 0 ]; then
    echo "Error: Original training file is empty."
    exit 1
fi

# Use awk for floating point calculation and ceiling for test size
TEST_LINES=$(awk -v total="${TOTAL_LINES}" -v ratio="${SPLIT_RATIO}" 'BEGIN { printf "%d\n", total * ratio + 0.999 }')
TRAIN_LINES=$((TOTAL_LINES - TEST_LINES))

echo "Total samples in ${ORIGINAL_TRAIN_FILE}: ${TOTAL_LINES}"
echo "Creating Test set (${TEST_FILE}) with ${TEST_LINES} samples (approx. ${SPLIT_RATIO}*100%)"
echo "Creating new Train set (${NEW_TRAIN_FILE}) with ${TRAIN_LINES} samples"

# --- Perform the split ---
# Shuffle the original file and split into test and train sets
shuf "${ORIGINAL_TRAIN_PATH}" | head -n "${TEST_LINES}" > "${TEST_PATH}"
shuf "${ORIGINAL_TRAIN_PATH}" | tail -n "${TRAIN_LINES}" > "${NEW_TRAIN_PATH}"

# Alternative using the same shuffle (might be slightly faster for large files)
# TEMP_SHUFFLED=$(mktemp)
# shuf "${ORIGINAL_TRAIN_PATH}" > "${TEMP_SHUFFLED}"
# head -n "${TEST_LINES}" "${TEMP_SHUFFLED}" > "${TEST_PATH}"
# tail -n "${TRAIN_LINES}" "${TEMP_SHUFFLED}" > "${NEW_TRAIN_PATH}"
# rm "${TEMP_SHUFFLED}"

# --- Make script executable by default ---
# chmod +x scripts/split_train_test.sh

echo "Splitting complete."
echo "New training file: ${NEW_TRAIN_PATH}"
echo "Test file: ${TEST_PATH}"

exit 0 