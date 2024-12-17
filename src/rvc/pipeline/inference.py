from pathlib import Path

import torch
import torchaudio
from scipy import signal

from rvc.model.rmvpe import RMVPE
from rvc.model.vits.models import SynthesizerTrnMs768NSFsid
from rvc.pipeline.preprocess import make_feature_extractor
from rvc.pipeline.utils import f0_coarse_representation


@torch.inference_mode()
def voice_conversion(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    model_path: Path,
) -> None:
    pitch_estimator = RMVPE(model_path / "rmvpe.pt").eval()
    feature_extractor = make_feature_extractor(model_path / "wav2vec2.pt")

    audio, sample_rate = torchaudio.load(input_path)
    resample = torchaudio.transforms.Resample(sample_rate, 16_000)
    audio = resample(audio)
    audio = audio.mean(0)  # convert to mono
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
    audio = signal.filtfilt(bh, ah, audio)

    # TODO split long audio on low amplitude timestamps (ideally based on vram)
    t_pad = 1 * 16_000
    # t_query = 6 * 16_000
    # t_center = 38 * 16_000
    # t_max = 41 * 16_000
    t_pad_target = 1 * 48_000
    window = 16_000 // 100
    # audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
    # low_amplitude_timestamps = []
    # if audio_pad.shape[0] > t_max:
    #     audio_sum = np.sum(audio_pad[t : t - window].abs() for t in range(window))
    #     for t in range(t_center, audio.shape[0], t_center):
    #         audio_sum_query = audio_sum[t - t_query : t + t_query]
    #         low_amplitude_timestamps.append(
    #             t - t_query + np.where(audio_sum_query == audio_sum_query.min())[0][0]
    #         )

    audio_adj = torch.nn.functional.pad(
        torch.tensor(audio.copy()).float().view(1, -1),
        (t_pad, t_pad),
        mode="reflect",
    )
    f0 = pitch_estimator(audio_adj)
    f0_coarse = f0_coarse_representation(f0)
    features = feature_extractor(audio_adj)

    # TODO use trained feature index

    features = torch.nn.functional.interpolate(
        features.permute(0, 2, 1), scale_factor=2
    ).permute(0, 2, 1)
    feature_max_len = min(features.size(1), audio_adj.size(1) // window)
    f0 = f0[:feature_max_len].view(1, -1)
    f0_coarse = f0_coarse[:feature_max_len].view(1, -1)
    features_lengths = torch.tensor([feature_max_len]).long()
    speaker_id = torch.zeros(1, dtype=torch.long)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]  # n_spk
    model = SynthesizerTrnMs768NSFsid(
        *checkpoint["config"],
        is_half=False,
    )
    del model.enc_q
    model.load_state_dict(checkpoint["weight"], strict=False)
    model.eval()

    converted_audio, _, _ = model.infer(
        features,
        features_lengths,
        f0_coarse,
        f0,
        speaker_id,
    )
    converted_audio = converted_audio.squeeze(0)[:, t_pad_target:-t_pad_target]
    torchaudio.save(
        uri=output_path,
        src=converted_audio,
        sample_rate=48_000,
    )
