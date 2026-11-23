#!/bin/bash
# Train the text2semantic model from scratch on SLURM, see train_from_scratch.py.
#
# Run it on the login node (do not sbatch it), it checks the config and submits itself. Arguments are Hydra
# overrides, passed on to train_from_scratch.py:
#
#     bash tools/llama/train_from_scratch.sh project=emilia_en_9ep
#
# A job is limited to 5 days, so CHAIN jobs are queued one after the other (--dependency=singleton). Each
# one resumes from the latest checkpoint in results/<project>/checkpoints, the jobs that are still queued
# when the training has finished are cancelled. Run the same command again to continue after a failure.
#
# Settings, in the environment:
#
#     NODES=1 GPUS=2 CHAIN=3 PARTITION=gpu-big TIME=5-00:00:00 MEM=500G CPUS_PER_GPU=8 MAIL_USER=you@example.org
#
# NODES=2 trains on 2 nodes with GPUS each. trainer.num_nodes is set from NODES, and unless it is given,
# trainer.accumulate_grad_batches is scaled so that an optimizer step stays at the 1M tokens of the config.
#
# Without internet on the compute nodes, run with WANDB_MODE=offline and upload the run afterwards with
# `wandb sync results/<project>/wandb/offline-run-*`.

set -euo pipefail

export PYTHON="${PYTHON:-}"                     # default: .venv of my-dual-ar-tts, two folders up
NODES="${NODES:-1}"
GPUS="${GPUS:-2}"                               # per node
CHAIN="${CHAIN:-3}"                             # jobs queued after each other, unused ones are cancelled
PARTITION="${PARTITION:-gpu-big}"
TIME="${TIME:-5-00:00:00}"                      # the maximum of gpu-big
MEM="${MEM:-500G}"                              # the data loader workers keep all protos in memory
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"               # data.num_workers (6) + 2
MAIL_USER="${MAIL_USER:-}"                      # default: the mail address SLURM has for your account

# Taken from the cluster setup of the other training runs
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# ----------------------------------------------------------------------------
# In the job: one process per GPU
# ----------------------------------------------------------------------------
if [[ -n "${TRAIN_IN_JOB:-}" ]]; then
    echo "$(date '+%F %T') job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: $SLURM_NTASKS tasks"
    export PYTHONUNBUFFERED=1
    export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"  # srun does not inherit --cpus-per-task since SLURM 22.05

    status=0
    srun --kill-on-bad-exit=1 "$PYTHON" tools/llama/train_from_scratch.py "$@" || status=$?
    echo "$(date '+%F %T') training exited with $status"

    # Finished: the queued jobs of the chain are not needed. After a failure they resume the training,
    # a job that hits the time limit is killed before it gets here
    if [[ $status -eq 0 ]]; then
        scancel --user="$USER" --name="$SLURM_JOB_NAME" --state=PENDING
    fi
    exit "$status"
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

# The config is written for 2 GPUs: 2 x batch_size 4 x accumulate 16 x 8192 = 1M tokens per optimizer step
world_size=$(( NODES * GPUS ))
overrides=("trainer.num_nodes=$NODES")
if [[ " $* " != *" trainer.accumulate_grad_batches="* ]]; then
    if (( 32 % world_size != 0 )); then
        echo "$world_size GPUs do not divide 32, pass trainer.accumulate_grad_batches" >&2
        exit 1
    fi
    overrides+=("trainer.accumulate_grad_batches=$(( 32 / world_size ))")
fi
set -- "${overrides[@]}" "$@"

# The config with the overrides, checked as at the start of the training. Prints the project, the last
# step and the tokens per optimizer step
read -r project max_steps tokens_per_step < <("$PYTHON" - "$world_size" "$@" <<'EOF' | tail -n 1
import importlib.util
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

spec = importlib.util.spec_from_file_location("train_from_scratch", "tools/llama/train_from_scratch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

gpus, overrides = int(sys.argv[1]), sys.argv[2:]
with initialize_config_dir(str(Path("fish_speech/configs").resolve()), version_base="1.3"):
    cfg = compose("text2semantic_pretrain", overrides=overrides)

module.check_config(cfg)
tokens = gpus * cfg.data.batch_size * cfg.trainer.accumulate_grad_batches * cfg.max_length
print(cfg.project, cfg.trainer.max_steps, tokens)
EOF
) || true  # read fails on empty input, handled below
if [[ -z "${tokens_per_step:-}" ]]; then
    echo "The config check failed, see above" >&2
    exit 1
fi

# Two jobs of one project would write the same checkpoints
name="train-$project"
if [[ -n "$(squeue --me --noheader --name="$name")" ]]; then
    echo "A job of $project is already queued or running:" >&2
    squeue --me --name="$name" >&2
    exit 1
fi

last="results/$project/checkpoints/spaced/$(printf 'step_%09d' "$max_steps").ckpt"
if [[ -f "$last" ]]; then
    echo "$project is finished, $last exists. Use a new project name for a new run" >&2
    exit 1
fi

mkdir -p logs
options=(
    --job-name="$name"
    --partition="$PARTITION"
    --time="$TIME"
    --nodes="$NODES"
    --ntasks-per-node="$GPUS"
    --gpus-per-node="$GPUS"
    --cpus-per-task="$CPUS_PER_GPU"
    --mem="$MEM"
    --dependency=singleton
    --chdir="$PWD"
    --output="logs/$name-%j.out"
    --error="logs/$name-%j.err"
    --mail-type=END,FAIL
    --export=ALL,TRAIN_IN_JOB=1
)
[[ -z "$MAIL_USER" ]] || options+=(--mail-user="$MAIL_USER")

jobs=()
for _ in $(seq "$CHAIN"); do
    job="$(sbatch --parsable "${options[@]}" "$script" "$@")"
    jobs+=("${job%%;*}")  # with several clusters, sbatch prints "id;cluster"
done

tokens="$(numfmt --grouping "$tokens_per_step" 2>/dev/null || echo "$tokens_per_step")"
echo "Submitted $project: jobs ${jobs[*]} on $NODES node(s) x $GPUS GPUs, $max_steps steps of $tokens tokens"
echo "Output in $(realpath -m "results/$project"), logs in logs/$name-<job>.out/.err"
