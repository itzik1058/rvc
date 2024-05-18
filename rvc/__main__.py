import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Optional

import faiss
import torch
from torch.amp import GradScaler
from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchaudio.models import wav2vec2_base
from torchaudio.transforms import MelScale, MelSpectrogram
from tqdm import trange
from typer import Argument, Option, Typer

from rvc.dataset import RVCBatch, RVCDataset
from rvc.model import commons
from rvc.model.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.model.models import MultiPeriodDiscriminatorV2, SynthesizerTrnMs768NSFsid
from rvc.rmvpe import RMVPE

logging.basicConfig()

app = Typer()


@app.command()
def train(
    checkpoint_path: Annotated[Path, Argument()] = Path("env/rvc.pt"),
    data_path: Annotated[Path, Argument()] = Path("env/data"),
    model_path: Annotated[Path, Argument()] = Path("env/models"),
    cache_path: Annotated[Optional[Path], Option()] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    rmvpe = RMVPE(model_path / "rmvpe.pt").eval()
    wav2vec2 = wav2vec2_base()
    wav2vec2.load_state_dict(torch.load(model_path / "wav2vec2.pt"))

    def feature_extractor(waveform: torch.Tensor) -> torch.Tensor:
        features, _ = wav2vec2.extract_features(waveform)
        return features[-1]

    dataset = RVCDataset(
        data_path,
        cache_path,
        pitch_estimator=rmvpe,
        feature_extractor=feature_extractor,
    )

    with torch.inference_mode():
        features = []
        for sample in dataset:
            features.append(sample.features)
        all_features = torch.concat(features)

        n_ivf = min(
            int(16 * math.sqrt(all_features.size(0))), all_features.size(0) // 39
        )
        index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(all_features)
        batch_size_add = 8192
        for i in range(0, all_features.size(0), batch_size_add):
            index.add(all_features[i : i + batch_size_add])
        faiss.write_index(
            index,
            checkpoint_path.with_suffix(".index").as_posix(),
        )

    data_loader = DataLoader(dataset, batch_size=4, collate_fn=RVCDataset.collate_fn)

    hps = SimpleNamespace(
        sample_rate=48000,
        data=SimpleNamespace(
            filter_length=2048,
            hop_length=480,
            max_wav_value=32768.0,
            mel_fmax=None,
            mel_fmin=0.0,
            n_mel_channels=128,
            sampling_rate=48000,
            win_length=2048,
        ),
        model=SimpleNamespace(
            filter_channels=768,
            gin_channels=256,
            hidden_channels=192,
            inter_channels=192,
            kernel_size=3,
            n_heads=2,
            n_layers=6,
            p_dropout=0,
            resblock="1",
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock_kernel_sizes=[3, 7, 11],
            spk_embed_dim=109,
            upsample_initial_channel=512,
            upsample_kernel_sizes=[24, 20, 4, 4],
            upsample_rates=[12, 10, 2, 2],
            use_spectral_norm=False,
            is_half=False,
        ),
        train=SimpleNamespace(
            batch_size=4,
            betas=[0.8, 0.99],
            c_kl=1.0,
            c_mel=45,
            epochs=20000,
            eps=1e-09,
            fp16_run=False,
            init_lr_ratio=1,
            learning_rate=0.0001,
            log_interval=200,
            lr_decay=0.999875,
            seed=1234,
            segment_size=17280,
            warmup_epochs=0,
        ),
    )

    net_g = SynthesizerTrnMs768NSFsid(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
        sr=hps.sample_rate,
    )
    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=hps.model.use_spectral_norm)
    optim_g = AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    net_g.load_state_dict(torch.load(model_path / "f0G48k.pth")["model"])
    net_d.load_state_dict(torch.load(model_path / "f0D48k.pth")["model"])
    scheduler_g = ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = ExponentialLR(optim_d, gamma=hps.train.lr_decay)
    scaler = GradScaler(enabled=False)
    mel_transform = MelScale(
        n_mels=hps.data.n_mel_channels,
        sample_rate=hps.data.sampling_rate,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_stft=hps.data.filter_length // 2 + 1,
    )
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        power=1,
    )
    for epoch in trange(20):
        net_g.train()
        net_d.train()
        for batch in data_loader:
            batch: RVCBatch
            features_adj = batch.features.repeat_interleave(2, 1)
            features_lengths_adj = batch.features_lengths * 2
            feature_max_len = features_adj.size(1)
            f0_adj = batch.f0[:, :feature_max_len]
            f0_coarse_adj = batch.f0_coarse[:, :feature_max_len]
            spectrogram_adj = batch.spectrogram[:, :, :feature_max_len]
            spectrogram_lengths_adj = torch.as_tensor(
                [min(length, feature_max_len) for length in batch.spectrogram_lengths]
            )
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(
                features_adj,
                features_lengths_adj,
                f0_coarse_adj,
                f0_adj,
                spectrogram_adj,
                spectrogram_lengths_adj,
                batch.speaker_id,
            )
            mel = mel_transform(spectrogram_adj)
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_transform(y_hat.float().squeeze(1))
            y_hat_mel = y_hat_mel[:, :, : y_mel.size(2)]  # FIXME match lengths
            wave = commons.slice_segments(
                batch.audio, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            loss_mel = l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

        scheduler_g.step()
        scheduler_d.step()

        print(
            f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
        )

    checkpoint = net_g.state_dict()
    opt = {}
    opt["weight"] = {}
    for key in checkpoint.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = checkpoint[key].half()
    opt["config"] = [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]
    opt["info"] = f"{epoch}epoch"
    opt["sr"] = hps.sample_rate
    opt["f0"] = 1
    opt["version"] = "v2"
    torch.save(opt, checkpoint_path)


@app.command()
@torch.inference_mode()
def inference(
    input_path: Annotated[Path, Argument()],
    model_path: Annotated[Path, Argument()] = Path("env/models"),
):
    pass
