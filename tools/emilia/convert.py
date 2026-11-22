"""Stream Emilia from Hugging Face and write Fish-Speech protobuf files, without storing any audio.

Emilia (amphion/Emilia-Dataset, gated) is a set of tars holding one mp3 and one json per clip. Every tar is
read over HTTP, its clips are decoded in memory and encoded with the Fish-Speech codec, and only the
transcripts and codes are written:

    <output>/train/EN-B000000.protos   one group per speaker, clips in their original order
    <output>/val/EN-B000000.protos     the same for held-out speakers
    <output>/flagged/EN-B000000.tsv    clips whose transcript matches --flag, if any (id, match, kept or the
                                       reason why it was dropped, transcript). Flagged clips are not dropped
    <output>/stats/EN-B000000.json     written last, marks the tar as finished

Speakers with fewer than --min-clips clips are dropped, because a training sequence only has one speaker.
The clips and speakers in excluded.txt, next to this script, are always left out.

Finished tars are skipped, so jobs can be resubmitted. Request access on the dataset page, log in with
`huggingface-cli login` and get the codec once:

    huggingface-cli download fishaudio/fish-speech-1.5 firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
        --local-dir checkpoints/fish-speech-1.5

Run it from the fish-speech folder. The tars are distributed over the tasks of a SLURM array, one GPU each:

    python tools/emilia/convert.py --output data/emilia_en
    python tools/emilia/convert.py --output data/emilia_en --exclude excluded.txt   # see search.py
    python tools/emilia/convert.py --output data/emilia_en --summary                # progress and totals

Emilia is licensed CC BY-NC 4.0.
"""

import hashlib
import io
import json
import os
import queue
import re
import sys
import tarfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import click
import numpy as np
import pyrootutils
import requests
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import HfApi, get_token, hf_hub_url
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.datasets.protos.text_data_pb2 import Semantics, Sentence, TextData
from fish_speech.datasets.protos.text_data_stream import pack_pb_stream

REPO = "amphion/Emilia-Dataset"
REVISION = "d7f2f7340a6385696f3766c8049fa920a4707c07"  # pinned, 2025-02-28

EXCLUDED = Path(__file__).with_name("excluded.txt")

# Regular expressions, not case sensitive. The transcripts are automatic, so names are often misspelled
DEFAULT_FLAGS = (
    r"\bdavid sl[aei]+y?t+[eo]rs?\b",  # David Slater, also Slatter
    r"\bdana [cs]h[ie]+v+\w*",  # Dana Chivvis, also Chivis, Chivas
    r"\bs[uo]l+a ?w[ae]+[sz]+[iy]\w*",  # Sulawesi, also Sula Wesi, Sulawasi
)


class HttpStream:
    """Sequential reader of a file on the Hub that reconnects where it was interrupted."""

    def __init__(self, path: str, revision: str = REVISION, max_retries: int = 12):
        self.url = hf_hub_url(REPO, path, repo_type="dataset", revision=revision)
        self.max_retries = max_retries
        self.position = 0
        self.size: Optional[int] = None
        self.response: Optional[requests.Response] = None

    def connect(self) -> None:
        self.close()
        headers = {"Range": f"bytes={self.position}-", "Accept-Encoding": "identity"}
        if token := get_token():
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(self.url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        if response.status_code != 206:
            # The server would restart from the beginning and corrupt the tar
            response.close()
            raise IOError(f"Range request not honoured: {response.status_code}")

        self.size = int(response.headers["Content-Range"].split("/")[1])
        self.response = response

    def read(self, size: int) -> bytes:
        for attempt in range(self.max_retries):
            try:
                if self.response is None:
                    self.connect()

                data = self.response.raw.read(size)
                if data or self.position >= self.size:
                    self.position += len(data)
                    return data

                raise IOError("Connection closed early")
            except requests.HTTPError as e:
                # No access or no such file: retrying does not help
                if e.response.status_code in (401, 403, 404):
                    raise
                error = e
            except Exception as e:
                error = e

            self.close()
            delay = min(60, 2**attempt)
            logger.warning(
                f"{error!r} at byte {self.position}, retrying in {delay} s"
            )
            time.sleep(delay)

        raise IOError(f"Gave up on {self.url} at byte {self.position}")

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
            self.response = None


def read_clips(path: str) -> Iterator[tuple[dict, bytes]]:
    """Yield the metadata and the mp3 of every clip of a tar."""

    stream = HttpStream(path)
    pending = {}

    try:
        with tarfile.open(fileobj=stream, mode="r|", bufsize=1 << 20) as tar:
            for member in tar:
                if not member.isfile():
                    continue

                key, extension = os.path.splitext(member.name)
                pending.setdefault(key, {})[extension] = tar.extractfile(member).read()

                if {".json", ".mp3"} <= pending[key].keys():
                    clip = pending.pop(key)
                    yield json.loads(clip[".json"]), clip[".mp3"]
    finally:
        stream.close()

    if pending:
        logger.warning(f"{path}: {len(pending)} clips without mp3 or json")


def clean_transcript(text: str) -> str:
    # Same as in tools/llama/build_dataset.py: no { xxx } and < xxx >
    text = re.sub(r"\{.*?\}", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def drop_reason(meta: dict, text: str, excluded: set[str]) -> Optional[str]:
    if meta["id"] in excluded or meta["speaker"] in excluded:
        return "excluded"

    if meta["language"] != "en":
        return "language"

    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return "no text"

    # Clips in other languages that were labelled as English
    if any("LATIN" not in unicodedata.name(c, "") for c in letters):
        return "not latin"

    return None


def is_validation(speaker: str, fraction: float) -> bool:
    digest = hashlib.sha256(speaker.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def decode(mp3: bytes) -> Optional[tuple[np.ndarray, int]]:
    try:
        wav, sample_rate = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
    except Exception:
        return None

    return (wav.mean(axis=1), sample_rate) if len(wav) > 0 else None


class Codec:
    def __init__(self, checkpoint: str, batch_seconds: float, device: str):
        from fish_speech.models.vqgan.inference import load_model

        self.model = load_model("firefly_gan_vq", checkpoint, device=device)
        self.sample_rate = self.model.spec_transform.sample_rate
        self.batch_seconds = batch_seconds
        self.device = device
        self.resamplers = {}

    def resample(self, wavs: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.sample_rate:
            return wavs

        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                sample_rate, self.sample_rate
            ).to(self.device)

        return self.resamplers[sample_rate](wavs)

    def batches(self, clips: list[tuple[np.ndarray, int]]) -> Iterator[list[int]]:
        """Indices of clips with the same sample rate and a similar length."""

        order = sorted(range(len(clips)), key=lambda i: (clips[i][1], -len(clips[i][0])))
        batch = []

        for i in order:
            if batch:
                longest, sample_rate = len(clips[batch[0]][0]), clips[batch[0]][1]
                seconds = (len(batch) + 1) * longest / sample_rate
                if clips[i][1] != sample_rate or seconds > self.batch_seconds:
                    yield batch
                    batch = []
            batch.append(i)

        if batch:
            yield batch

    @torch.inference_mode()
    def encode(self, clips: list[tuple[np.ndarray, int]]) -> list[np.ndarray]:
        """Encode clips, given as (samples, sample rate), into codes of shape [8, frames]."""

        codes = [None] * len(clips)

        for batch in self.batches(clips):
            sample_rate = clips[batch[0]][1]
            lengths = [len(clips[i][0]) for i in batch]
            padded = np.zeros((len(batch), max(lengths)), dtype=np.float32)
            for row, i in zip(padded, batch):
                row[: len(clips[i][0])] = clips[i][0]

            wavs = self.resample(torch.from_numpy(padded).to(self.device), sample_rate)
            # Length after resampling, as computed by torchaudio
            lengths = torch.tensor(
                [-(-n * self.sample_rate // sample_rate) for n in lengths],
                device=self.device,
            )
            indices, frames = self.model.encode(wavs[:, None], lengths)
            indices, frames = indices.cpu().numpy(), frames.cpu().numpy()

            for row, i, n in zip(indices, batch, frames):
                codes[i] = row[:, :n].astype(np.uint16)

        return codes


def load_audio(
    path: str,
    excluded: set[str],
    pool: ThreadPoolExecutor,
    buffer_size: int,
    stats: Counter,
    flag: Optional[re.Pattern],
    flagged: dict[str, list[str]],
) -> Iterator[list[tuple[dict, str, tuple[np.ndarray, int]]]]:
    """Read, filter and decode a tar in the background, yield buffers of clips."""

    buffers = queue.Queue(maxsize=2)
    stop = threading.Event()

    def put(item) -> None:
        while not stop.is_set():
            try:
                return buffers.put(item, timeout=1)
            except queue.Full:
                continue

    def drop(meta: dict, reason: str) -> None:
        stats[f"drop: {reason}"] += 1
        if meta["id"] in flagged:
            flagged[meta["id"]][1] = reason

    def resolve(tasks: list) -> list:
        clips = []
        for meta, text, future in tasks:
            if (audio := future.result()) is None:
                drop(meta, "unreadable mp3")
            else:
                clips.append((meta, text, audio))
        return clips

    def run() -> None:
        try:
            tasks = []
            for meta, mp3 in read_clips(path):
                if stop.is_set():
                    return

                stats["clips in tar"] += 1
                text = clean_transcript(meta["text"])
                if flag is not None and (match := flag.search(text)):
                    flagged[meta["id"]] = [match[0], "kept", text]

                if reason := drop_reason(meta, text, excluded):
                    drop(meta, reason)
                    continue

                tasks.append((meta, text, pool.submit(decode, mp3)))
                if len(tasks) >= buffer_size:
                    put(resolve(tasks))
                    tasks = []

            put(resolve(tasks))
            put(None)
        except BaseException as e:
            put(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    try:
        while (item := buffers.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


def write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def convert_tar(
    path: str,
    output: Path,
    codec: Codec,
    pool: ThreadPoolExecutor,
    excluded: set[str],
    val_fraction: float,
    min_clips: int,
    buffer_size: int,
    flag: Optional[re.Pattern],
) -> dict:
    name, begin = Path(path).stem, time.time()
    stats, speakers = Counter(), defaultdict(list)
    flagged = {}

    for clips in load_audio(
        path, excluded, pool, buffer_size, stats, flag, flagged
    ):
        for (meta, text, audio), codes in zip(
            clips, codec.encode([audio for _, _, audio in clips])
        ):
            if codes.shape[1] == 0:
                stats["drop: too short"] += 1
                if meta["id"] in flagged:
                    flagged[meta["id"]][1] = "too short"
                continue

            speakers[meta["speaker"]].append((meta["id"], text, codes))

    data = {"train": [], "val": []}
    for speaker, clips in sorted(speakers.items()):
        # All clips of a speaker are in the same tar
        if len(clips) < min_clips:
            stats["drop: few clips"] += len(clips)
            stats["dropped speakers"] += 1
            for id, _, _ in clips:
                if id in flagged:
                    flagged[id][1] = "few clips"
            continue

        split = "val" if is_validation(speaker, val_fraction) else "train"
        clips.sort(key=lambda x: x[0])
        data[split].append(
            pack_pb_stream(
                TextData(
                    source="emilia",
                    name=speaker,
                    sentences=[
                        Sentence(
                            texts=[text],
                            semantics=[Semantics(values=x) for x in codes.tolist()],
                        )
                        for _, text, codes in clips
                    ],
                )
            )
        )
        stats[f"{split} speakers"] += 1
        stats[f"{split} clips"] += len(clips)
        stats[f"{split} frames"] += sum(codes.shape[1] for _, _, codes in clips)

    for split, groups in data.items():
        if groups:
            write_atomically(output / split / f"{name}.protos", b"".join(groups))

    if flagged:
        stats["flagged clips"] = len(flagged)
        rows = ["\t".join([id, *row]) for id, row in sorted(flagged.items())]
        data = "\n".join(rows) + "\n"
        write_atomically(output / "flagged" / f"{name}.tsv", data.encode())

    stats["seconds"] = round(time.time() - begin)
    write_atomically(
        output / "stats" / f"{name}.json",
        json.dumps({"tar": path, **stats}, indent=2).encode(),
    )

    return stats


def list_tars(subset: str) -> list[str]:
    files = HfApi().list_repo_tree(
        REPO, subset, repo_type="dataset", revision=REVISION, recursive=True
    )
    return sorted(f.path for f in files if f.path.endswith(".tar"))


def print_summary(output: Path, num_tars: int) -> None:
    files = sorted((output / "stats").glob("*.json"))
    total = Counter()
    for file in files:
        stats = json.loads(file.read_text())
        total.update({k: v for k, v in stats.items() if isinstance(v, int)})

    print(f"{len(files)} of {num_tars} tars finished")
    for key, value in sorted(total.items()):
        hours = f"  ({value * 2048 / 44100 / 3600:.1f} h)" if "frames" in key else ""
        print(f"{key:24s} {value:14,d}{hours}")

    for file in sorted((output / "flagged").glob("*.tsv")):
        print(file.read_text(encoding="utf-8"), end="")


@click.command(help=__doc__.split("\n")[0])
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--subset", default="Emilia/EN", help="Folder of the dataset repository")
@click.option("--tar", "only", multiple=True, help="Convert only these, e.g. EN-B001139")
@click.option(
    "--exclude",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    help="File with a clip or speaker id per line, in addition to excluded.txt",
)
@click.option(
    "--checkpoint-path",
    default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
)
@click.option("--val-fraction", default=0.001, help="Share of held-out speakers")
@click.option(
    "--min-clips",
    default=5,
    help="Drop speakers with fewer clips (after the other filters)",
)
@click.option(
    "--flag",
    multiple=True,
    default=DEFAULT_FLAGS,
    help="Record clips whose transcript matches this regular expression",
)
@click.option("--batch-seconds", default=600.0, help="Padded audio per codec batch")
@click.option("--buffer-size", default=512, help="Clips sorted by length together")
@click.option("--num-workers", default=8, help="Threads decoding mp3")
@click.option(
    "--num-shards", default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
)
@click.option(
    "--shard-id",
    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0)),
)
@click.option("--device", default="cuda")
@click.option("--summary", is_flag=True, help="Print the progress and exit")
def main(
    output: Path,
    subset: str,
    only: tuple[str, ...],
    exclude: tuple[Path, ...],
    checkpoint_path: str,
    val_fraction: float,
    min_clips: int,
    flag: tuple[str, ...],
    batch_seconds: float,
    buffer_size: int,
    num_workers: int,
    num_shards: int,
    shard_id: int,
    device: str,
    summary: bool,
):
    tars = list_tars(subset)
    if len(tars) == 0:
        raise click.ClickException(f"No tars in {subset}")

    if summary:
        return print_summary(output, len(tars))

    if not 0 <= shard_id < num_shards:
        raise click.ClickException(f"Shard {shard_id} of {num_shards} does not exist")

    if only:
        tars = [t for t in tars if Path(t).stem in only]

    tars = tars[shard_id::num_shards]
    todo = [t for t in tars if not (output / "stats" / f"{Path(t).stem}.json").exists()]
    logger.info(f"Shard {shard_id}/{num_shards}: {len(todo)} of {len(tars)} tars to do")

    if len(todo) == 0:
        return

    excluded = set()
    for file in (EXCLUDED, *exclude):
        lines = file.read_text(encoding="utf-8").splitlines()
        excluded |= {x.split()[0] for x in lines if x.strip() and x[0] != "#"}
    logger.info(f"{len(excluded)} excluded clips and speakers")

    pattern = "|".join(f"(?:{x})" for x in flag if x)
    pattern = re.compile(pattern, re.IGNORECASE) if pattern else None
    codec = Codec(checkpoint_path, batch_seconds, device)
    failed = []

    with ThreadPoolExecutor(num_workers) as pool:
        for i, path in enumerate(todo):
            try:
                stats = convert_tar(
                    path,
                    output,
                    codec,
                    pool,
                    excluded,
                    val_fraction,
                    min_clips,
                    buffer_size,
                    pattern,
                )
            except requests.HTTPError as e:
                if e.response.status_code in (401, 403):
                    raise click.ClickException(
                        f"No access to {REPO}: request it on the dataset page and log in"
                    )
                logger.exception(f"{path} failed")
                failed.append(path)
                continue
            except Exception:
                logger.exception(f"{path} failed")
                failed.append(path)
                continue

            hours = (stats["train frames"] + stats["val frames"]) * 2048 / 44100 / 3600
            logger.info(
                f"[{i + 1}/{len(todo)}] {Path(path).stem}: {hours:.1f} h of audio "
                f"in {stats['seconds'] / 60:.1f} min"
            )

    if failed:
        logger.error(f"{len(failed)} tars failed, run the same command again: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
