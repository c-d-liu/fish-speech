"""Train the Fish-Speech 1.5 text2semantic model from scratch.

Only the tokenizer of the released checkpoint is needed, the weights are randomly initialised:

    huggingface-cli download fishaudio/fish-speech-1.5 tokenizer.tiktoken \
        --local-dir checkpoints/fish-speech-1.5

Start the training from the fish-speech folder. All settings are in
fish_speech/configs/text2semantic_pretrain.yaml and can be overridden here:

    python tools/llama/train_from_scratch.py project=my_run \
        "train_protos=[data/train_protos]" "val_protos=[data/val_protos]"

Training resumes from the latest training state in results/<project>/checkpoints, so use a new project
name for every new run. Weights are kept at evenly spaced steps in checkpoints/spaced, and for the lowest
validation loss in checkpoints/best. To load one, extract it into a copy of the model folder, which the
script fills with the config and the tokenizer:

    cp -r results/my_run/model results/my_run/model_1000
    python tools/extract_model.py results/my_run/checkpoints/spaced/step_000001000.ckpt \
        results/my_run/model_1000/model.pth
"""

from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from lightning.pytorch.utilities import rank_zero_only

from fish_speech.train import train
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)


def find_protos(paths: list[str]) -> set[Path]:
    files = set()
    for pattern in paths:
        for path in map(Path, braceexpand(pattern)):
            found = [path] if path.is_file() else [*path.rglob("*.proto*")]
            files.update(i.resolve() for i in found if i.suffix in (".proto", ".protos"))

    return files


def check_config(cfg: DictConfig) -> None:
    """Fail before the training starts, not in a data loading worker."""

    if not Path(cfg.tokenizer_path).is_file():
        raise FileNotFoundError(
            f"No tokenizer at {cfg.tokenizer_path}, "
            "the download command is at the top of this script"
        )

    if isinstance(cfg.train_protos, str) or isinstance(cfg.val_protos, str):
        raise ValueError('Pass the protos as lists: "train_protos=[path]"')

    train_files = find_protos(cfg.train_protos)
    val_files = find_protos(cfg.val_protos)
    log.info(f"Found {len(train_files)} training and {len(val_files)} validation files")

    for name, files in (("train_protos", train_files), ("val_protos", val_files)):
        if len(files) == 0:
            raise FileNotFoundError(f"No .protos files in {name}: {cfg[name]}")

    if overlap := train_files & val_files:
        raise ValueError(f"Used for both training and validation: {sorted(overlap)[0]}")

    if cfg.max_length > cfg.model.model.config.max_seq_len:
        raise ValueError("max_length is larger than the max_seq_len of the model")

    if cfg.trainer.max_steps % cfg.num_checkpoints != 0:
        raise ValueError("trainer.max_steps must be a multiple of num_checkpoints")

    # Fewer files than loaders: some files are read by several loaders, see split_by_rank_worker
    if len(train_files) < cfg.data.num_workers:
        log.warning(
            f"Only {len(train_files)} training files for {cfg.data.num_workers} workers "
            "per GPU, use smaller shards or fewer workers"
        )


@rank_zero_only
def save_model_folder(cfg: DictConfig) -> None:
    """Write the config and tokenizer needed to load the extracted weights."""

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    config = hydra.utils.instantiate(cfg.model.model.config)

    num_tokens = max(tokenizer.all_special_tokens_with_ids.values()) + 1
    if num_tokens > config.vocab_size:
        raise ValueError(f"{num_tokens} tokens, but vocab_size is {config.vocab_size}")

    path = Path(cfg.paths.run_dir) / "model"
    tokenizer.save_pretrained(path)
    config.save(path / "config.json")


@hydra.main(
    version_base="1.3",
    config_path="../../fish_speech/configs",
    config_name="text2semantic_pretrain",
)
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    save_model_folder(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
