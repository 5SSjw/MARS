#!/bin/bash

# Configuration
EVAL_SCRIPT="/work/xinyu/EAGLE/eagle/data_2/calculate_bleu.py"
REFERENCE="/work/xinyu/EAGLE/eagle/data_2/wmt19/question.jsonl"
SUMMARY_FILE="wmt19_evaluation_summary.txt"

# Clear previous summary
echo "Evaluation Summary - $(date)" > "$SUMMARY_FILE"
echo "==================================================" >> "$SUMMARY_FILE"
printf "%-60s | %-10s | %-10s\n" "File" "BLEU" "chrF" >> "$SUMMARY_FILE"
echo "--------------------------------------------------" >> "$SUMMARY_FILE"

# Function to evaluate a single file
evaluate_file() {
    local file_path="$1"
    local model_name="$2" 
    local filename=$(basename "$file_path")
    
    # Run evaluation and capture output
    output=$(python3 "$EVAL_SCRIPT" --reference "$REFERENCE" --generated "$file_path" 2>&1)
    
    # Extract scores - looking for lines like "BLEU: 29.67" or "chrF: 59.61"
    # match colon followed by space and then the number
    bleu=$(echo "$output" | grep "^BLEU" | grep -oE ": [0-9]+\.[0-9]+" | head -n 1 | awk '{print $2}')
    chrf=$(echo "$output" | grep "^chrF" | grep -oE ": [0-9]+\.[0-9]+" | head -n 1 | awk '{print $2}')
    
    # Prepend model name to filename
    display_name="${model_name}/${filename}"
    
    printf "%-60s | %-10s | %-10s\n" "$display_name" "$bleu" "$chrf" >> "$SUMMARY_FILE"
    echo "Processed $display_name: BLEU=$bleu, chrF=$chrf"
}

# Directories to search
DIRS=(
    # "/work/xinyu/EAGLE/eagle/rebuttal/qwen3-8b"
    # "/work/xinyu/EAGLE/eagle/rebuttal/qwen3-32b"
    # "/work/xinyu/EAGLE1/eagle/rebuttal/qwen3-8b"
    # "/work/xinyu/EAGLE1/eagle/rebuttal/qwen3-32b"
    "/work/xinyu/EAGLE/eagle/rebuttal/vicuna13b"
    "/work/xinyu/EAGLE1/eagle/rebuttal/vicuna13b"
)

# Iterate through directories
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Scanning directory: $dir"
        
        # Determine base name for display
        if [[ "$dir" == *"/EAGLE/eagle/rebuttal"* ]]; then
            base="EAGLE"
        elif [[ "$dir" == *"/EAGLE1/eagle/rebuttal"* ]]; then
            base="EAGLE1"
        else
            base="UNKNOWN"
        fi
        
        subdir=$(basename "$dir")
        model="${base}/${subdir}"
        
        # Find jsonl files matching wmt19 pattern
        find "$dir" -name "wmt19*.jsonl" | sort | while read -r file; do
            evaluate_file "$file" "$model"
        done
    else
        echo "Directory not found: $dir"
    fi
done

echo ""
echo "Evaluation complete. Summary saved to $SUMMARY_FILE"
cat "$SUMMARY_FILE"
