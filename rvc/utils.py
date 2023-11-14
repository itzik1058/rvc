from pathlib import Path

import ffmpeg
import numpy as np
import numpy.typing as npt
from pydub import AudioSegment


def load_audio(path: Path, sampling_rate_hz: int) -> npt.NDArray[np.float32]:
    stdout, _ = (
        ffmpeg.input(path, threads=0)
        .output(
            "-",
            format="f32le",
            acodec="pcm_f32le",
            ac=1,
            ar=sampling_rate_hz,
        )
        .run(
            cmd=["ffmpeg", "-nostdin"],
            capture_stdout=True,
            capture_stderr=True,
        )
    )
    return np.frombuffer(stdout, np.float32).flatten()
