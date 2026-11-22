#!/bin/bash
# Convert Emilia to Fish-Speech protobuf files as a SLURM array, see convert.py.
#
# Run it on the login node (do not sbatch it), it checks the setup and submits itself:
#
#     bash tools/emilia/convert.sh
#
# Run the same command again to continue after a timeout or a failure, finished tars are skipped. Arguments are
# passed on to convert.py, the settings below can be changed in the environment:
#
#     OUTPUT=data/emilia_test NUM_SHARDS=1 bash tools/emilia/convert.sh --tar EN-B001014
#
# Progress: python tools/emilia/convert.py --output data/emilia_en --summary
# Logs:     logs/emilia_<job>_<task>.log, the totals at the end are in logs/emilia_summary_<job>.log

#SBATCH --job-name=emilia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export OUTPUT="${OUTPUT:-data/emilia_en}"
export NUM_SHARDS="${NUM_SHARDS:-8}"        # array tasks, each with one GPU. Fixed, so that resubmitting keeps the shards
export PYTHON="${PYTHON:-}"                 # default: .venv of my-dual-ar-tts, two folders up
PARTITION="${PARTITION:-gpu-big}"
TIME="${TIME:-24:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-$NUM_SHARDS}" # GPUs used at the same time
export CHECKPOINT="${CHECKPOINT:-checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth}"

# ----------------------------------------------------------------------------
# In the job: one shard of the tars
# ----------------------------------------------------------------------------
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "$(date '+%F %T') task $SLURM_ARRAY_TASK_ID of $NUM_SHARDS on $(hostname)"
    export PYTHONUNBUFFERED=1

    for attempt in 1 2 3; do
        status=0
        "$PYTHON" tools/emilia/convert.py \
            --output "$OUTPUT" \
            --checkpoint-path "$CHECKPOINT" \
            --num-shards "$NUM_SHARDS" \
            --shard-id "$SLURM_ARRAY_TASK_ID" \
            --num-workers "$(( SLURM_CPUS_PER_TASK > 3 ? SLURM_CPUS_PER_TASK - 2 : 1 ))" \
            "$@" || status=$?

        # 0: finished, 1: some tars failed and are worth another attempt, others: a problem with the setup
        [[ $status -eq 1 ]] || exit "$status"
        echo "$(date '+%F %T') attempt $attempt failed"
        [[ $attempt -eq 3 ]] || sleep "${RETRY_DELAY:-300}"
    done
    exit 1
fi

# ----------------------------------------------------------------------------
# On the login node: check and submit
# ----------------------------------------------------------------------------
script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$script")/../.."

if [[ -z "$PYTHON" ]]; then
    PYTHON="$(cd ../.. && pwd)/.venv/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "No Python at $PYTHON, set PYTHON" >&2
    exit 1
fi

# Two jobs working on the same tars would overwrite each other's files
if [[ -n "$(squeue --me --noheader --name=emilia)" ]]; then
    echo "An emilia job is already queued or running:" >&2
    squeue --me --name=emilia >&2
    exit 1
fi

"$PYTHON" tools/emilia/convert.py --output "$OUTPUT" --checkpoint-path "$CHECKPOINT" --check

mkdir -p logs "$OUTPUT"
job="$(sbatch --parsable \
    --array="0-$(( NUM_SHARDS - 1 ))%$MAX_PARALLEL" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --chdir="$PWD" \
    --output="logs/emilia_%A_%a.log" \
    "$script" "$@")"
job="${job%%;*}"  # with several clusters, sbatch prints "id;cluster"

# Runs when all tasks have ended, also after failures
printf -v summary '%q ' "$PYTHON" tools/emilia/convert.py --output "$OUTPUT" --summary
sbatch --parsable \
    --dependency="afterany:$job" \
    --job-name=emilia-summary \
    --partition="${SUMMARY_PARTITION:-cpu}" \
    --cpus-per-task=1 \
    --mem=8G \
    --time=00:30:00 \
    --chdir="$PWD" \
    --output="logs/emilia_summary_$job.log" \
    --wrap "$summary" > /dev/null

echo "Submitted job $job: $NUM_SHARDS tasks, output in $(realpath "$OUTPUT")"
