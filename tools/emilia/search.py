"""Search the sources and transcripts of the English Emilia data, e.g. for a podcast to exclude from training.

Three searches, none of which downloads audio:

    # 1. Transcripts (3.1 GB of metadata are streamed once, the English part is cached, 1.3 GB)
    python tools/emilia/search.py transcripts --cache emilia_en.tsv.gz \
        --pattern "this american life" --pattern "ira glass" --output hits.tsv

    # 2. Overlap with known transcripts, e.g. of the stimuli of an experiment: utterances sharing a sequence of
    #    --ngram words with a text file. The most reliable test for specific recordings
    python tools/emilia/search.py overlap --cache emilia_en.tsv.gz --output overlap.tsv transcripts/*.txt

    # 3. The list of source URLs. It is in another gated repository (amphion/Emilia), for which access has to be
    #    requested separately
    python tools/emilia/search.py sources --pattern "this american life" --pattern thisamericanlife

The results of 1 and 2 have the clip id in the first column, so they can be passed to convert.py --exclude. By
default only the clips are listed, --speakers lists the whole speakers (all clips of a voice in a recording).
"""

import csv
import gzip
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, NamedTuple

import click
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import convert

# The old layout of the dataset, with the metadata of all clips in one archive
METADATA = "openemilia_all.tar.gz"
METADATA_REVISION = "fc71e07e8572f5f3be1dbd02ed3172a4d298f152"


class Clip(NamedTuple):
    id: str
    duration: float
    text: str

    @property
    def speaker(self) -> str:
        return self.id.rsplit("_", 1)[0]


def build_cache(cache: Path) -> None:
    stream = convert.HttpStream(METADATA, revision=METADATA_REVISION)
    temporary = cache.with_name(cache.name + ".tmp")
    found = False

    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as out:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                name = member.name.split("/")[-1]
                if not (member.isfile() and re.fullmatch(r"EN_B\d+\.jsonl", name)):
                    if found:
                        # Sorted by language, nothing English follows
                        break
                    continue

                found = True
                for line in tar.extractfile(member):
                    clip = json.loads(line)
                    text = " ".join(clip["text"].split())
                    out.write(f"{clip['id']}\t{clip['duration']}\t{text}\n")

                click.echo(f"{name} ({stream.position / 1e9:.2f} GB)", err=True)

    os.replace(temporary, cache)


def read_clips(cache: Path) -> Iterator[Clip]:
    if not cache.exists():
        build_cache(cache)

    with gzip.open(cache, "rt", encoding="utf-8") as f:
        for line in f:
            id, duration, text = line.rstrip("\n").split("\t", 2)
            yield Clip(id, float(duration), text)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", "").replace("’", ""))


def write_hits(output: Path, hits: list[tuple[Clip, str]], speakers: bool) -> None:
    lines, seen = [], set()
    for clip, why in hits:
        id = clip.speaker if speakers else clip.id
        if id not in seen:
            seen.add(id)
            lines.append(f"{id}\t{why}\t{clip.text}\n")

    output.write_text("".join(lines), encoding="utf-8")
    click.echo(f"{len(lines)} {'speakers' if speakers else 'clips'} in {output}")


@click.group(help=__doc__.split("\n")[0])
def main():
    pass


cache_option = click.option(
    "--cache", type=click.Path(path_type=Path), default="emilia_en.tsv.gz"
)
output_option = click.option(
    "--output", type=click.Path(path_type=Path), required=True
)
speakers_option = click.option(
    "--speakers", is_flag=True, help="List the speakers instead of the clips"
)
pattern_option = click.option(
    "--pattern", multiple=True, required=True, help="Regular expression, not case sensitive"
)


@main.command()
@cache_option
@output_option
@speakers_option
@pattern_option
def transcripts(cache: Path, output: Path, speakers: bool, pattern: tuple[str, ...]):
    """Search for regular expressions in the transcripts."""

    regex = re.compile("|".join(f"(?:{p})" for p in pattern), re.IGNORECASE)
    hits = [(c, m[0]) for c in read_clips(cache) if (m := regex.search(c.text))]
    write_hits(output, hits, speakers)

    for clip, why in hits[:50]:
        click.echo(f"{clip.id}  [{why}]  {clip.text[:150]}")


@main.command()
@cache_option
@output_option
@speakers_option
@click.option("--ngram", default=8, help="Consecutive words that have to be the same")
@click.argument(
    "files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
def overlap(cache: Path, output: Path, speakers: bool, ngram: int, files: tuple):
    """Search for clips that share word sequences with text files."""

    sources = {}
    for file in files:
        w = words(file.read_text(encoding="utf-8"))
        for i in range(len(w) - ngram + 1):
            sources.setdefault(tuple(w[i : i + ngram]), file.name)
    click.echo(f"{len(sources)} sequences of {ngram} words in {len(files)} files")

    # Looking up the first two words is much cheaper than building all sequences
    starts = {s[:2] for s in sources}
    hits, per_file, seconds = [], Counter(), defaultdict(float)
    for clip in read_clips(cache):
        w = words(clip.text)
        for i in range(len(w) - ngram + 1):
            if (w[i], w[i + 1]) in starts and (
                name := sources.get(tuple(w[i : i + ngram]))
            ):
                hits.append((clip, f"{name}: {' '.join(w[i:i + ngram])}"))
                per_file[name] += 1
                seconds[name] += clip.duration
                break

    write_hits(output, hits, speakers)
    for name, n in per_file.most_common():
        click.echo(f"{name}: {n} clips, {seconds[name] / 60:.1f} min")


@main.command()
@pattern_option
@click.option("--file", default="emilia_101k_source_meta.csv")
def sources(pattern: tuple[str, ...], file: str):
    """Search the list of source URLs (amphion/Emilia)."""

    path = hf_hub_download("amphion/Emilia", file, repo_type="dataset")
    regex = re.compile("|".join(f"(?:{p})" for p in pattern), re.IGNORECASE)

    csv.field_size_limit(2**31 - 1)
    rows = matches = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for rows, row in enumerate(csv.reader(f)):
            if rows == 0:
                click.echo("columns: " + ", ".join(row))
            elif regex.search("\t".join(row)):
                matches += 1
                click.echo("\t".join(x[:200] for x in row))

    click.echo(f"{matches} of {rows} sources match")


if __name__ == "__main__":
    main()
