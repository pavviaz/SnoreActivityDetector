import torch
import torch.nn as nn

from meta_dicts import FEATURES


def ms2samples(ms, sample_rate):
    return int(ms * sample_rate / 1000)


def eq(inp, target_dbfs):
    inputs_ndim = len(inp.shape)
    rms = torch.sqrt(torch.mean(inp**2, dim=inputs_ndim - 1, keepdim=True))
    rms = torch.clamp(rms, 1e-10, 1e10)

    wave_dBFS = 20 * torch.log10(rms)
    dBFS_diff = target_dbfs - wave_dBFS

    return inp * (10 ** (dBFS_diff / 20))


def identity(inp, **kwargs):
    return inp


class ModelFeatured(nn.Module):
    def __init__(self, model, data_cfg):
        super(ModelFeatured, self).__init__()
        self.norm = eq if data_cfg.general.use_equalization else identity

        self.fe = FEATURES[data_cfg.general.feature_type]["cls"](
            sample_rate=data_cfg.general.sample_rate,
            win_length=ms2samples(
                data_cfg.general.win_len, data_cfg.general.sample_rate
            ),
            n_fft=data_cfg.general.n_fft,
            hop_length=ms2samples(
                data_cfg.general.hop_len, data_cfg.general.sample_rate
            ),
            n_mels=data_cfg.general.n_mels,
        )

        self.main_model = model
        self.main_model.eval()

        self.data_cfg = data_cfg

    def forward(self, inputs):
        norm_inp = self.norm(inputs, self.data_cfg.general.use_equalization)

        features = self.fe(norm_inp)

        return self.main_model(features)
