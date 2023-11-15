import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from rvc.rmvpe.constants import (
    MEL_FMAX,
    MEL_FMIN,
    N_CLASS,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from rvc.rmvpe.deepunet import DeepUnet
from rvc.rmvpe.seq import BiGRU


class E2E(nn.Module):
    def __init__(
        self,
        hop_length: int,
        n_blocks: int,
        n_gru: int,
        kernel_size: int | tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super(E2E, self).__init__()
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            win_length=WINDOW_LENGTH,
            hop_length=hop_length,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX,
            n_mels=N_MELS,
        )
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
