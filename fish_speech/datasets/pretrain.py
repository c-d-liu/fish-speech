import random
from typing import Iterator, Optional

import torch
from torch.distributed import get_rank, is_initialized
from torch.utils.data import get_worker_info

from fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech.datasets.semantic import AutoTextSemanticInstructionIterableDataset
from fish_speech.text.clean import clean_text
from fish_speech.tokenizer import FishTokenizer

SYSTEM_PROMPT = "Speak out the provided text."
SKIP_TEXT = "<|skip_text|>"


class PretrainDataset(AutoTextSemanticInstructionIterableDataset):
    """
    Dataset for training from scratch. A sequence is one conversation: the system prompt,
    followed by consecutive utterances (transcript, audio) of one speaker.

    The parent class was written for finetuning. The differences, chosen to match the
    released 1.5 model where its weights show how it was trained:

    1. The transcript is a target of the text head (transcript_loss). The output rows of
       text tokens are trained in the released model. Skipped transcripts are no target.
    2. <|voice|> is not a target, the released model was never trained to predict it.
    3. The system prompt is given once, as in inference, and not for every utterance.
    4. Every utterance can be sampled. The parent takes a window of max_length // 20
       utterances, of which the collator only keeps the first few, so the last
       utterances of a speaker are never used. Here the start is uniform over the
       utterances, which are added (wrapping around at the end) until max_length.
    5. Codebook labels are only set where there is a frame to predict. The parent also
       sets them to 0 at the end of an utterance, where the fast transformer is not run:
       these positions have no gradient and only add ln(1024) to the average.
    """

    def __init__(
        self,
        proto_files: list[str],
        tokenizer: FishTokenizer,
        max_length: int = 8192,
        num_codebooks: int = 8,
        skip_text_prob: float = 0.0,
        transcript_loss: bool = True,
        fixed_samples: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            proto_files: .protos files or folders containing them
            tokenizer: tokenizer
            max_length: length of a sequence, longer ones are cut
            num_codebooks: number of codebooks
            skip_text_prob: probability to replace a transcript by <|skip_text|>
            transcript_loss: train the text head on the transcript
            fixed_samples: give the same samples in every epoch, for validation
            seed: seed for the order of the files, and for the samples if they are fixed
        """

        super().__init__(
            proto_files=list(proto_files),
            seed=seed,
            max_length=max_length,
            tokenizer=tokenizer,
            num_codebooks=num_codebooks,
            skip_text_prob=skip_text_prob,
        )

        self.transcript_loss = transcript_loss
        self.fixed_samples = fixed_samples

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random

        if self.fixed_samples:
            rank = get_rank() if is_initialized() else 0
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            rng = random.Random(f"{self.seed}-{rank}-{worker_id}")

        while True:
            yield self.augment(rng)

    def encode(
        self, utterances: list[tuple[str, torch.Tensor, bool]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode utterances, given as (transcript, codes, skip_text)."""

        messages = [Message(role="system", parts=[TextPart(text=SYSTEM_PROMPT)])]

        for text, codes, skip_text in utterances:
            messages += [
                Message(
                    role="user",
                    parts=[TextPart(text=SKIP_TEXT if skip_text else text)],
                    cal_loss=self.transcript_loss and not skip_text,
                ),
                # The modality adds <|voice|> to the header, which is no target
                Message(
                    role="assistant",
                    parts=[VQPart(codes=codes)],
                    modality="voice",
                    cal_loss=True,
                ),
            ]

        encoded = Conversation(messages=messages).encode(tokenizer=self.tokenizer)
        codes = torch.cat(encoded.vq_parts, dim=1).long()
        shape = (self.num_codebooks + 1, len(encoded.tokens))

        tokens = torch.full(shape, CODEBOOK_PAD_TOKEN_ID, dtype=torch.long)
        tokens[0] = encoded.tokens
        tokens[1:, encoded.vq_mask_tokens] = codes

        labels = torch.full(shape, -100, dtype=torch.long)
        labels[0] = encoded.labels
        labels[1:, encoded.vq_mask_labels] = codes

        return tokens, labels

    def augment(self, rng=random) -> Optional[dict[str, torch.Tensor]]:
        if self.groups is None:
            self.init_mock_data_server()

        group = rng.choices(self.groups, weights=self.group_weights, k=1)[0]
        num_sentences = len(group.sentences)
        begin = rng.randrange(num_sentences)

        utterances, num_frames = [], 0

        for i in range(num_sentences):
            sentence = group.sentences[(begin + i) % num_sentences]
            text = clean_text(rng.choice(sentence.texts))
            codes = torch.tensor(
                [list(x.values) for x in sentence.semantics], dtype=torch.int
            )

            if codes.size(0) != self.num_codebooks:
                raise ValueError(
                    f"Expected {self.num_codebooks} codebooks, got {codes.size(0)}"
                )

            utterances.append((text, codes, rng.random() < self.skip_text_prob))
            num_frames += codes.size(1)

            # The frames alone fill the sequence
            if num_frames >= self.max_length:
                break

        tokens, labels = self.encode(utterances)
        tokens = tokens[:, : self.max_length]
        labels = labels[:, : self.max_length]

        # The fast transformer reads the target frame from the next input column, which
        # does not exist for the last position
        labels[1:, -1] = -100

        return {"tokens": tokens, "labels": labels}
