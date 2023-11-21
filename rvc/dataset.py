import math
from pathlib import Path
from typing import Callable, SupportsIndex, TypedDict
from fairseq.models.hubert import HubertModel

import pydub.silence
import scipy.signal
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import Dataset

from rvc.utils import SAMPLE_RATE, numpy_to_pydub, pydub_to_numpy

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


class RVCSample(TypedDict):
    audio: torch.Tensor
    f0: torch.Tensor
    f0_coarse: torch.Tensor
    features: torch.Tensor


class RVCDataset(Dataset[RVCSample]):
    def __init__(
        self,
        path: Path,
        cache_path: Path,
        pitch_estimator: Callable[[torch.Tensor], torch.Tensor],
        hubert: HubertModel,
        min_silence_ms: int = 500,
        silence_thresh_dbfs: int = -42,
        keep_silence_ms: int = 500,
        seek_step_ms: int = 15,
        min_sample_ms: int = 1500,
        max_sample_ms: int = 3000,
        sample_overlap_ms: int = 300,
    ) -> None:
        super().__init__()

        self.pitch_estimator = pitch_estimator
        self.feature_extractor = hubert
        self.min_silence_ms = min_silence_ms
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.keep_silence_ms = keep_silence_ms
        self.seek_step_ms = seek_step_ms
        self.min_sample_ms = min_sample_ms
        self.max_sample_ms = max_sample_ms
        self.sample_overlap_ms = sample_overlap_ms

        self.samples: list[Path] = []
        self._load(path, cache_path)

    def __getitem__(self, index: SupportsIndex) -> RVCSample:
        path = self.samples[index]
        audio, _ = torchaudio.load(path)
        return {
            "audio": audio,
            "f0": torch.load(path.with_suffix("f0")),
            "f0_coarse": torch.load(path.with_suffix("f0c")),
            "features": torch.load(path.with_suffix("ft")),
        }

    def __len__(self) -> int:
        return len(self.samples)

    @torch.no_grad()
    def _load(self, path: Path, cache_path: Path) -> None:
        for p in path.iterdir():
            audio, sample_rate = torchaudio.load(p)

            b, a = scipy.signal.butter(
                N=5,
                Wn=48,
                btype="high",
                fs=sample_rate,
            )
            resample = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)

            audio = audio.mean(0)  # 1 channel
            audio = scipy.signal.lfilter(b, a, audio).astype("float32")

            segments = pydub.silence.split_on_silence(
                numpy_to_pydub(audio, sample_rate),
                min_silence_len=self.min_silence_ms,
                silence_thresh=self.silence_thresh_dbfs,
                keep_silence=self.keep_silence_ms,
                seek_step=self.seek_step_ms,
            )
            for idx, segment in enumerate(segments):
                end = 0
                while end < len(segment):
                    start = max(0, end - self.sample_overlap_ms)
                    end = min(start + self.max_sample_ms, len(segment))
                    if end - start < self.min_sample_ms:
                        continue

                    sample = torch.tensor(
                        pydub_to_numpy(segment[start:end]).reshape(1, -1)
                    )
                    # smooth with normalized sample
                    sample = sample.lerp(sample / sample.abs().max() * 0.9, 0.75)
                    sample = resample(sample)
                    segment_path = cache_path / f"{p.stem}_{idx}_{start}_{end}.wav"
                    torchaudio.save(
                        segment_path,
                        sample,
                        SAMPLE_RATE,
                        format="wav",
                    )

                    f0 = self.pitch_estimator(sample)
                    torch.save(f0, segment_path.with_suffix(".f0"))

                    f0_mel = 1127 * torch.log(1 + f0 / 700)
                    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * (
                        F0_BIN - 2
                    ) / (F0_MEL_MAX - F0_MEL_MIN) + 1
                    f0_mel[f0_mel <= 1] = 1
                    f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
                    f0_coarse = f0_mel.round().int()
                    torch.save(f0_coarse, segment_path.with_suffix(".f0c"))

                    logits, _ = self.feature_extractor.extract_features(
                        source=sample,
                        padding_mask=torch.zeros_like(sample).bool(),
                        output_layer=12,
                    )
                    logits = logits.squeeze(0)
                    torch.save(logits, segment_path.with_suffix(".ft"))

                    self.samples.append(segment_path)
