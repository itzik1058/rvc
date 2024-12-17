from pathlib import Path

import torch
from torchaudio.models import wav2vec2_base


def make_feature_extractor(path: Path):
    wav2vec2 = wav2vec2_base()
    wav2vec2.load_state_dict(torch.load(path, weights_only=True))

    def feature_extractor(waveform: torch.Tensor) -> torch.Tensor:
        features, _ = wav2vec2.extract_features(waveform)
        return features[-1]

    return feature_extractor
