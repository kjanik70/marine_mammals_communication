#!/bin/bash
# Run SanctSound DAC pipeline with a pool of N parallel workers.
# Automatically launches new deployments as others complete.
# Restarts each process every MAX_FLACS_PER_RUN to prevent memory leaks.

MAX_PARALLEL=2
MAX_FLACS_PER_RUN=50
LOGDIR="runs"
SCRIPT="scripts/process_sanctsound_humpback.py"
OUTPUT_DIR="data/tokenized/sanctsound_humpback_dac"

# Deployments to process (station deployment_num expected_total)
# hi05_01 (78/78) and hi04_01 (17/17) complete — all others resume from .done files
DEPLOYMENTS=(
  "hi03 3 513"
  "hi04 3 512"
  "hi01 2 501"
  "hi04 2 452"
  "hi01 1 451"
  "hi01 3 192"
  "hi03 1 49"
)

declare -A PIDS       # pid -> "station dep total"
declare -A RUN_COUNT  # "station_dep" -> run count

is_deployment_done() {
  local station=$1 dep=$2 total=$3
  local done_file="${OUTPUT_DIR}/.done_${station}_$(printf '%02d' $dep).txt"
  if [ -f "$done_file" ]; then
    local done_count
    done_count=$(wc -l < "$done_file")
    [ "$done_count" -ge "$total" ]
  else
    return 1
  fi
}

launch_deployment() {
  local station=$1 dep=$2 total=$3
  local dep_key="${station}_${dep}"
  local run_num=${RUN_COUNT[$dep_key]:-0}
  run_num=$((run_num + 1))
  RUN_COUNT[$dep_key]=$run_num
  local logfile="${LOGDIR}/sanctsound_dac_${station}_${dep}.log"

  PYTHONPATH=. python3 -u "$SCRIPT" \
    --codec dac --save-2d \
    --station "$station" --deployment "$dep" \
    --whale-cv-threshold 0.8 --energy-ratio-threshold 0.4 --min-whale-rms 0.01 \
    --max-flacs-per-run "$MAX_FLACS_PER_RUN" \
    >> "$logfile" 2>&1 &

  local pid=$!
  PIDS[$pid]="$station $dep $total"
  local done_count=0
  local done_file="${OUTPUT_DIR}/.done_${station}_$(printf '%02d' $dep).txt"
  [ -f "$done_file" ] && done_count=$(wc -l < "$done_file")
  echo "[$(date '+%H:%M:%S')] Launched ${station^^}_$(printf '%02d' $dep) run #${run_num} (PID $pid, ${done_count}/${total} done)"
}

# Launch initial batch
idx=0
for ((i=0; i<MAX_PARALLEL && idx<${#DEPLOYMENTS[@]}; i++)); do
  read -r station dep total <<< "${DEPLOYMENTS[$idx]}"
  if is_deployment_done "$station" "$dep" "$total"; then
    echo "[$(date '+%H:%M:%S')] SKIP: ${station^^}_$(printf '%02d' $dep) already complete"
    idx=$((idx + 1))
    i=$((i - 1))  # don't count this as a launched worker
    continue
  fi
  launch_deployment "$station" "$dep" "$total"
  idx=$((idx + 1))
  sleep 5  # stagger launches to avoid simultaneous DAC loading
done

echo "[$(date '+%H:%M:%S')] Pool started with $MAX_PARALLEL workers, ${#DEPLOYMENTS[@]} total deployments"
echo ""

# Wait for processes, launch new ones as slots free up
while [ ${#PIDS[@]} -gt 0 ]; do
  for pid in "${!PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      # Process finished
      wait "$pid"
      exit_code=$?
      dep_info="${PIDS[$pid]}"
      unset PIDS[$pid]

      read -r station dep total <<< "$dep_info"
      dep_str="${station^^}_$(printf '%02d' $dep)"

      if [ $exit_code -eq 0 ]; then
        # Check if deployment is fully done or needs another run
        if is_deployment_done "$station" "$dep" "$total"; then
          echo "[$(date '+%H:%M:%S')] COMPLETE: $dep_str"
        else
          # Relaunch same deployment (periodic restart for memory management)
          local_done=0
          done_file="${OUTPUT_DIR}/.done_${station}_$(printf '%02d' $dep).txt"
          [ -f "$done_file" ] && local_done=$(wc -l < "$done_file")
          echo "[$(date '+%H:%M:%S')] RESTART: $dep_str (${local_done}/${total} done, memory reset)"
          sleep 3
          launch_deployment "$station" "$dep" "$total"
          continue  # don't launch from queue — we're restarting the same deployment
        fi
      else
        echo "[$(date '+%H:%M:%S')] FAILED:   $dep_str (exit $exit_code)"
      fi

      # Launch next deployment if any remain in queue
      while [ $idx -lt ${#DEPLOYMENTS[@]} ]; do
        read -r next_station next_dep next_total <<< "${DEPLOYMENTS[$idx]}"
        idx=$((idx + 1))
        if is_deployment_done "$next_station" "$next_dep" "$next_total"; then
          echo "[$(date '+%H:%M:%S')] SKIP: ${next_station^^}_$(printf '%02d' $next_dep) already complete"
          continue
        fi
        sleep 3
        launch_deployment "$next_station" "$next_dep" "$next_total"
        break
      done
    fi
  done
  sleep 10  # check every 10 seconds
done

echo ""
echo "[$(date '+%H:%M:%S')] All deployments complete!"

# Final counts
echo ""
echo "=== Final Status ==="
for f in "${OUTPUT_DIR}"/.done_*.txt; do
  dep=$(basename "$f" | sed 's/.done_//;s/.txt//')
  count=$(wc -l < "$f")
  echo "  $dep: $count FLACs processed"
done
npy_count=$(find "$OUTPUT_DIR" -name "*.npy" | wc -l)
echo ""
echo "Total .npy files: $npy_count"
