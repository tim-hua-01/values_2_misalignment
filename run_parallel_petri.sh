#!/bin/bash

# Parallel Petri evaluation orchestrator with tmux
# Cycles through 4 target models for each value pair
# Runs evaluations + grading with semaphore-based parallelism
#
# Usage:
#   ./run_parallel_petri.sh <input_csv> [--max-windows N] [--session-name NAME] [--output OUTPUT_CSV]
#
# CSV format (required columns: value1, value2, csv_source):
#   value1,value2,csv_source,auditor,judge,epochs,max_turns

set -euo pipefail

# Default values
MAX_WINDOWS=8
SESSION_NAME=""
INPUT_CSV=""
OUTPUT_CSV="final_petri_results.csv"

# Target models to cycle through
TARGET_MODELS=(
    "openai/gpt-4.1-mini"
    "openrouter/google/gemini-2.5-pro"
    "openrouter/anthropic/claude-3.5-sonnet"
    "openrouter/x-ai/grok-4"
)

MODEL_SHORTNAMES=(
    "gpt41"
    "gemini25"
    "claude35"
    "grok4"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-windows)
            MAX_WINDOWS="$2"
            shift 2
            ;;
        --session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT_CSV" ]; then
                INPUT_CSV="$1"
            else
                echo "Error: Unknown argument '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$INPUT_CSV" ]; then
    echo "Usage: $0 <input_csv> [--max-windows N] [--session-name NAME] [--output OUTPUT_CSV]"
    echo ""
    echo "Options:"
    echo "  --max-windows N    Maximum concurrent tmux windows (default: 8)"
    echo "  --session-name     Custom tmux session name"
    echo "  --output           Output CSV file path (default: final_petri_results.csv)"
    echo ""
    echo "Example:"
    echo "  $0 value_pairs.csv --max-windows 12 --output my_results.csv"
    exit 1
fi

if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input CSV '$INPUT_CSV' not found"
    exit 1
fi

if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed"
    exit 1
fi

# Set default session name if not provided
if [ -z "$SESSION_NAME" ]; then
    SESSION_NAME="petri_$(date +%Y%m%d_%H%M%S)"
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Error: tmux session '$SESSION_NAME' already exists"
    echo "Attach: tmux attach -t $SESSION_NAME"
    echo "Kill: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create directories
mkdir -p logs
mkdir -p intermediate_results

# Validate input CSV before proceeding
echo "Validating input CSV..."
uv run python validate_parallel_input.py "$INPUT_CSV"

# Capture start time
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")

echo "=========================================="
echo "Parallel Petri Evaluation Orchestrator"
echo "=========================================="
echo "Start time: $START_TIME_HUMAN"
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"
echo "Session name: $SESSION_NAME"
echo "Max concurrent windows: $MAX_WINDOWS"
echo ""

# Function to sanitize value names for filenames
sanitize_name() {
    echo "$1" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//'
}

# Build job queue
echo "Building job queue..."

# Create temporary file for job queue
JOB_QUEUE_FILE="/tmp/petri_job_queue_$$.txt"
> "$JOB_QUEUE_FILE"  # Clear/create file

NUM_VALUE_PAIRS=0

while IFS=, read -r value1 value2 csv_source auditor judge epochs max_turns || [ -n "$value1" ]; do
    # Remove quotes
    value1=$(echo "$value1" | sed 's/^"//;s/"$//')
    value2=$(echo "$value2" | sed 's/^"//;s/"$//')
    csv_source=$(echo "$csv_source" | sed 's/^"//;s/"$//')
    auditor=$(echo "$auditor" | sed 's/^"//;s/"$//')
    judge=$(echo "$judge" | sed 's/^"//;s/"$//')
    epochs=$(echo "$epochs" | sed 's/^"//;s/"$//')
    max_turns=$(echo "$max_turns" | sed 's/^"//;s/"$//')
    
    # Skip if required fields missing
    if [ -z "$value1" ] || [ -z "$value2" ] || [ -z "$csv_source" ]; then
        continue
    fi
    
    # Create 4 jobs (one per model)
    for i in "${!TARGET_MODELS[@]}"; do
        target="${TARGET_MODELS[$i]}"
        model_short="${MODEL_SHORTNAMES[$i]}"
        
        # Build job string: value1|value2|csv_source|target|model_short|auditor|judge|epochs|max_turns
        job="$value1|$value2|$csv_source|$target|$model_short|$auditor|$judge|$epochs|$max_turns"
        echo "$job" >> "$JOB_QUEUE_FILE"
    done
    
    NUM_VALUE_PAIRS=$((NUM_VALUE_PAIRS + 1))
done < <(tail -n +2 "$INPUT_CSV")

TOTAL_JOBS=$(wc -l < "$JOB_QUEUE_FILE" | tr -d ' ')
echo "Created $TOTAL_JOBS jobs ($NUM_VALUE_PAIRS value pairs × 4 models)"
echo ""

# Create tmux session
tmux new-session -d -s "$SESSION_NAME"

# Get the first window number (respects user's base-index setting)
FIRST_WINDOW=$(tmux list-windows -t "$SESSION_NAME" -F "#{window_index}" | head -1)

# Failure log (shared across windows). Any failed job appends a line here.
FAILURE_LOG="logs/${SESSION_NAME}_FAILURES.txt"
> "$FAILURE_LOG"

# Create job runner script
JOB_RUNNER_SCRIPT="/tmp/petri_job_runner_$$.sh"
cat > "$JOB_RUNNER_SCRIPT" << 'RUNNER_EOF'
#!/bin/bash
set -euo pipefail

# Parse job string
IFS='|' read -r value1 value2 csv_source target model_short auditor judge epochs max_turns <<< "$1"
failure_log="$2"

# Sanitize names for filenames
v1_safe=$(echo "$value1" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//')
v2_safe=$(echo "$value2" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_//;s/_$//')
timestamp=$(date +%H%M%S)

# Generate filenames
LOG_NAME="logs/${v1_safe}_vs_${v2_safe}_${model_short}_${timestamp}.eval"
INTERMEDIATE_CSV="intermediate_results/${v1_safe}_${v2_safe}_${model_short}.csv"

echo "=========================================="
echo "Job: $value1 vs $value2 (model: $model_short)"
echo "=========================================="
echo "Target: $target"
echo "CSV Source: $csv_source"
echo "Log: $LOG_NAME"
echo ""

echo "Running evaluation..."
cmd=(uv run python petri_base_script.py --target "$target" --value1 "$value1" --value2 "$value2" --csv "$csv_source" --log-name "$LOG_NAME")
if [ -n "${auditor:-}" ]; then
    cmd+=(--auditor "$auditor")
fi
if [ -n "${judge:-}" ]; then
    cmd+=(--judge "$judge")
fi
if [ -n "${epochs:-}" ]; then
    cmd+=(--epochs "$epochs")
fi
if [ -n "${max_turns:-}" ]; then
    cmd+=(--max-turns "$max_turns")
fi
"${cmd[@]}"

echo ""
echo "Grading results..."

# Run grading script
uv run python petri_grade_script.py \
    --log "$LOG_NAME" \
    --csv "$csv_source" \
    --output "$INTERMEDIATE_CSV" \
    --value1 "$value1" \
    --value2 "$value2"

echo ""
echo "Job completed successfully!"
echo "Log: $LOG_NAME"
echo "Results: $INTERMEDIATE_CSV"
echo ""
echo "Closing window..."
sleep 1
exit 0
RUNNER_EOF

chmod +x "$JOB_RUNNER_SCRIPT"

# Function to run jobs with semaphore
run_jobs_with_semaphore() {
    local job_file="$JOB_QUEUE_FILE"
    local active_windows=0
    local completed_jobs=0
    local window_index=0
    
    echo "Starting job execution..."
    echo ""
    
    # Read jobs into array (portable way without mapfile)
    local jobs=()
    while IFS= read -r line; do
        jobs+=("$line")
    done < "$job_file"
    
    local total_jobs=${#jobs[@]}
    local first_window=true
    
    for job in "${jobs[@]}"; do
        # Fail fast: if any prior job failed, stop launching new jobs.
        if [ -s "$FAILURE_LOG" ]; then
            echo ""
            echo "ERROR: At least one job failed. See: $FAILURE_LOG"
            echo "Leaving tmux session '$SESSION_NAME' running for inspection: tmux attach -t $SESSION_NAME"
            exit 1
        fi

        # Wait if we're at max capacity
        while [ $active_windows -ge $MAX_WINDOWS ]; do
            sleep 2
            # Count active windows
            active_windows=$(tmux list-windows -t "$SESSION_NAME" 2>/dev/null | wc -l | tr -d ' ')
        done
        
        # Increment window index for display
        window_index=$((window_index + 1))
        window_name="job_$window_index"
        
        # Create new window and run job
        if [ "$first_window" = true ]; then
            # First window already exists, rename it
            tmux rename-window -t "$SESSION_NAME:$FIRST_WINDOW" "$window_name"
            tmux send-keys -t "$SESSION_NAME:$window_name" "cd $(pwd)" C-m
            # Run job; on failure, leave window open and mark FAILED
            tmux send-keys -t "$SESSION_NAME:$window_name" "bash \"$JOB_RUNNER_SCRIPT\" '$job' \"$FAILURE_LOG\"; exit_code=\$?; if [ \$exit_code -eq 0 ]; then tmux kill-window -t \"$SESSION_NAME:$window_name\"; else echo \"JOB FAILED (exit=\$exit_code)\"; echo \"${window_name} | \$job\" >> \"$FAILURE_LOG\"; tmux rename-window -t \"$SESSION_NAME:$window_name\" \"FAILED_${window_name}\"; exec bash; fi" C-m
            first_window=false
        else
            # Create new window
            tmux new-window -t "$SESSION_NAME" -n "$window_name"
            tmux send-keys -t "$SESSION_NAME:$window_name" "cd $(pwd)" C-m
            # Run job; on failure, leave window open and mark FAILED
            tmux send-keys -t "$SESSION_NAME:$window_name" "bash \"$JOB_RUNNER_SCRIPT\" '$job' \"$FAILURE_LOG\"; exit_code=\$?; if [ \$exit_code -eq 0 ]; then tmux kill-window -t \"$SESSION_NAME:$window_name\"; else echo \"JOB FAILED (exit=\$exit_code)\"; echo \"${window_name} | \$job\" >> \"$FAILURE_LOG\"; tmux rename-window -t \"$SESSION_NAME:$window_name\" \"FAILED_${window_name}\"; exec bash; fi" C-m
        fi
        
        active_windows=$((active_windows + 1))
        
        # Extract value names for display
        IFS='|' read -r v1 v2 _ _ model_short _ <<< "$job"
        echo "Started job $window_index/$total_jobs: $v1 vs $v2 ($model_short)"
        
        sleep 1  # Small delay to avoid overwhelming tmux
    done
    
    echo ""
    echo "All jobs launched. Waiting for completion..."
    
    # Wait for all windows to finish
    while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
        # If any job failed, stop waiting and exit non-zero (leave session running).
        if [ -s "$FAILURE_LOG" ]; then
            echo ""
            echo "ERROR: At least one job failed. See: $FAILURE_LOG"
            echo "Leaving tmux session '$SESSION_NAME' running for inspection: tmux attach -t $SESSION_NAME"
            exit 1
        fi
        active_windows=$(tmux list-windows -t "$SESSION_NAME" 2>/dev/null | wc -l | tr -d ' ')
        if [ $active_windows -eq 0 ]; then
            break
        fi
        sleep 5
    done
    
    echo ""
    echo "All jobs completed!"
}

# Run the orchestration
run_jobs_with_semaphore

# Cleanup temporary files
rm -f "$JOB_RUNNER_SCRIPT"
rm -f "$JOB_QUEUE_FILE"

echo ""
echo "=========================================="
echo "Orchestration Complete!"
echo "=========================================="
echo "Logs saved to: logs/"
echo "Intermediate results: intermediate_results/"
echo ""

# Aggregate results
echo "Aggregating results..."
uv run python aggregate_petri_results.py --input-dir intermediate_results/ --output "$OUTPUT_CSV"

# Check if aggregation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Aggregation successful! Cleaning up intermediate results..."
    rm -rf intermediate_results/*
    echo "Intermediate results cleaned up."
    # Calculate and display timing
    END_TIME=$(date +%s)
    END_TIME_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo "=========================================="
    echo "Pipeline Complete!"
    echo "=========================================="
    echo "Final results saved to: $OUTPUT_CSV"
    echo "Logs saved to: logs/"
    echo ""
    echo "Start time:  $START_TIME_HUMAN"
    echo "End time:    $END_TIME_HUMAN"
    printf "Duration:    %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
else
    echo ""
    echo "ERROR: Aggregation failed. Keeping intermediate results for inspection."
    exit 1
fi
