import numpy as np
import numpy.typing as npt
from pydub import AudioSegment

SAMPLE_RATE = 16_000
TARGET_SAMPLE_RATE = 48_000


def numpy_to_pydub(audio: npt.NDArray[np.float_], sample_rate: int) -> AudioSegment:
    audio_f32 = audio.astype(np.float32)
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
