import math
import warnings

import numpy as np
import numpy.typing as npt
import torch
from pydub import AudioSegment

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


def numpy_to_pydub(audio: npt.NDArray[np.float64], sample_rate: int) -> AudioSegment:
    audio_f32 = audio.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio_pcm32 = (audio_f32 * (2**31 - 1)).round().astype(np.int32)
    return AudioSegment(
        audio_pcm32.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_pcm32.dtype.itemsize,
        channels=1,
    )


def pydub_to_numpy(segment: AudioSegment) -> npt.NDArray[np.float32]:
    array = np.array(
        segment.get_array_of_samples(),
        dtype=np.float32,
    ).reshape((-1, segment.channels))
    array /= 1 << (8 * segment.sample_width - 1)
    return array


def f0_coarse_representation(f0: torch.Tensor) -> torch.Tensor:
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_coarse = f0_mel
    f0_coarse[f0_coarse > 0] = 1 + (F0_BIN - 2) * (
        f0_coarse[f0_coarse > 0] - F0_MEL_MIN
    ) / (F0_MEL_MAX - F0_MEL_MIN)
    f0_coarse[f0_coarse <= 1] = 1
    f0_coarse[f0_coarse > F0_BIN - 1] = F0_BIN - 1
    f0_coarse = f0_coarse.round().int()
    return f0_coarse
