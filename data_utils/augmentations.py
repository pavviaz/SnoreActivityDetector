import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from julius import lowpass_filter


def prob2bool(prob):
    return (
        prob
        if isinstance(prob, bool)
        else np.random.choice([True, False], p=[prob, 1 - prob])
    )


def ms2samples(ms, sample_rate):
    return int(ms * sample_rate / 1000)


class AugSequence(torch.nn.Module):
    def __init__(self, augmenters):
        super().__init__()
        self.augmenters = torch.nn.Sequential(*augmenters)

    def forward(self, inputs):
        return self.augmenters(inputs)


class Augmenter(torch.nn.Module):
    def __init__(self, probability, sample_rate=16000):
        super().__init__()
        self.probability = max(0, min(1.0, probability))
        self.sample_rate = sample_rate

    def set_probability(self, p):
        self.probability = p


class Impulse(Augmenter):
    def __init__(
        self,
        impulse_path,
        level=(0.3, 0.9),
        probability=1.0,
        noise_max_samples=22050,
        device="cpu",
    ):
        super().__init__(probability)
        self.device = device
        self.noise_max_samples = noise_max_samples
        self.impulse = self.load_impulse(impulse_path, noise_max_samples)
        self.level = level

    def load_impulse(self, path, max_samples):
        impulse, _ = torchaudio.load(path, num_frames=max_samples)

        impulse = (
            torch.mean(impulse, dim=0, keepdim=False)
            if impulse.shape[0] > 1
            else impulse
        )

        if impulse.shape[-1] < max_samples:
            impulse = F.pad(impulse, (0, max_samples - impulse.shape[-1]))

        return torch.unsqueeze(torch.unsqueeze(impulse, 0), 0).to(self.device)

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        bs = inputs.shape[0]
        impulse_level = np.random.uniform(
            self.level[0], self.level[1], (bs, 1, 1)
        ).astype(np.float32)
        impulse_level = torch.tensor(impulse_level, device=self.device)

        inputs_expanded = torch.unsqueeze(inputs, 0)
        inputs_expanded = F.pad(inputs_expanded, (0, self.noise_max_samples - 1, 0, 0))
        output = F.conv2d(
            inputs_expanded,
            self.impulse,
            stride=(1, 1),
            padding="valid",
        )[0]
        output = output * impulse_level + inputs * (1 - impulse_level)

        # output /= (torch.max(output) + 0.00001)

        return output


class BackgroundNoise(Augmenter):
    def __init__(
        self,
        noise_path,
        batch_size,
        noise_max_samples,
        level=(0.05, 0.3),
        probability=1.0,
        device="cpu",
    ):
        super().__init__(probability)
        # self.noise_list = self.load_noise(noise_path, batch_size, noise_max_samples)
        self.noises = noise_path
        self.noise_max_samples = noise_max_samples
        self.level = level
        self.device = device

    def load_noise(self, path, batch_size, max_samples):
        def _load(path, max_samples):
            impulse, _ = torchaudio.load(path, num_frames=max_samples)

            impulse = (
                torch.mean(impulse, dim=0, keepdim=False)
                if impulse.shape[0] > 1
                else impulse
            )

            if impulse.shape[-1] < max_samples:
                impulse = F.pad(impulse, (0, max_samples - impulse.shape[-1]))

            return impulse.numpy()

        if not os.path.exists(path):
            raise IOError(f"Noise folder on {path} has not been found")

        noise_list = os.listdir(path)
        np.random.shuffle(noise_list)
        return np.array(
            [
                _load(os.path.join(path, el), max_samples)
                for el in noise_list[:batch_size]
            ]
        )

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        bs = inputs.shape[0]
        # assert bs == self.noise_list.shape[0], "bs"

        noise_level = np.random.uniform(
            self.level[0], self.level[1], (bs, 1, 1)
        ).astype(np.float32)

        self.noise_list = self.load_noise(self.noises, bs, self.noise_max_samples)
        np.random.shuffle(self.noise_list)
        noise_list = torch.tensor(self.noise_list, device=self.device)

        noise_ratios = torch.max(inputs, dim=-1)[0] / (
            torch.max(noise_list, dim=-1)[0] + 0.000001
        )
        noise = (
            noise_list
            * torch.unsqueeze(noise_ratios, -1)
            * torch.tensor(noise_level, device=self.device)
        )

        output = inputs + noise

        # output /= (torch.max(output) + 0.00001)

        return output


class GaussianNoise(Augmenter):
    def __init__(self, level=(0.05, 0.3), probability=1.0, device="cpu"):
        super().__init__(probability)
        self.level = level
        self.device = device

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        noise_level = np.random.uniform(self.level[0], self.level[1])

        noise = torch.randn(*inputs.shape, dtype=torch.float32, device=self.device)
        output = inputs + noise * noise_level

        return output


class Gain(Augmenter):
    def __init__(self, min_gain=-20.0, max_gain=-1, probability=1.0, device="cpu"):
        super().__init__(probability)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.device = device

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        gain = np.random.uniform(self.min_gain, self.max_gain)
        return torchaudio.transforms.Vol(gain, gain_type="db")(inputs)


class EqualizeAmplitude(Augmenter):
    def __init__(self, target_dBFS=-20, probability=1.0, device="cpu"):
        super().__init__(probability)
        self.target_dBFS = target_dBFS
        self.device = device

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        inputs_ndim = len(inputs.shape)
        rms = torch.sqrt(torch.mean(inputs**2, dim=inputs_ndim - 1, keepdim=True))
        rms = torch.clamp(rms, 1e-10, 1e10)

        wave_dBFS = 20 * torch.log10(rms)
        dBFS_diff = self.target_dBFS - wave_dBFS

        return inputs * (10 ** (dBFS_diff / 20))


class LowPassFilter(Augmenter):
    def __init__(
        self, min_cutoff_freq=1500, max_cutoff_freq=8000, probability=1.0, device="cpu"
    ):
        super().__init__(probability)
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.device = device

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        cutoff_freq = np.random.uniform(self.min_cutoff_freq, self.max_cutoff_freq)
        cutoff_freq /= self.sample_rate
        return lowpass_filter(inputs, cutoff_freq)


class TrimSilence(Augmenter):
    def __init__(
        self,
        win_len,
        hop_len,
        offset,
        loudest_chunk_amount,
        probability=1.0,
        device="cpu",
    ):
        super().__init__(probability)
        self.win_size = ms2samples(win_len, self.sample_rate)
        self.hop_len = ms2samples(hop_len, self.sample_rate)
        self.offset = ms2samples(offset, self.sample_rate)
        self.loudest_chunk_amount = loudest_chunk_amount

        self.fade = torchaudio.transforms.Fade(
            fade_in_len=self.sample_rate,
            fade_out_len=self.sample_rate,
            fade_shape="linear",
        )

        self.device = device

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        if len(inputs.shape) == 2:
            inputs = torch.unsqueeze(inputs, dim=0)

        audio_abs = torch.abs(torch.squeeze(inputs))
        loudest_chunks = []

        # window loop
        for idx in range(len(audio_abs)):
            l_bound = idx * self.hop_len
            r_bound = l_bound + self.win_size
            if r_bound > len(audio_abs):
                break
            mean_chunk = torch.mean(audio_abs[l_bound:r_bound])
            if len(loudest_chunks) < self.loudest_chunk_amount:
                loudest_chunks.append((mean_chunk, l_bound + self.win_size // 2))
            elif (t := min(loudest_chunks, key=lambda x: x[0]))[0] < mean_chunk:
                loudest_chunks.remove(t)
                loudest_chunks.append((mean_chunk, l_bound + self.win_size // 2))

        loudest_chunks = sorted(loudest_chunks, key=lambda x: x[1])
        clean_snore = torch.cat(
            [
                self.fade(
                    inputs[
                        :,
                        :,
                        el[-1]
                        - min(
                            abs(self.offset - loudest_chunks[0][1]), self.offset
                        ) : el[-1]
                        + self.offset,
                    ]
                )
                for el in loudest_chunks
            ],
            dim=-1,
        )

        return clean_snore


class TimeShift(Augmenter):
    def __init__(self, shift_factor=(0.1, 0.6), probability=1.0, device="cpu"):
        super().__init__(probability)
        self.shift_factor = shift_factor
        self.device = device

    def forward(self, inputs):
        if not prob2bool(self.probability):
            return inputs

        bs, _, length = inputs.shape

        outputs = torch.zeros_like(inputs)
        for i in range(bs):
            shift = (
                np.random.uniform(self.shift_factor[0], self.shift_factor[1]) * length
            )
            shift = np.random.choice([-1, 1]) * int(shift)

            outputs[i] = torch.roll(inputs[i], shift, dims=-1)

            if shift > 0:
                fade = torchaudio.transforms.Fade(
                    fade_in_len=min(self.sample_rate, length - shift)
                )
                outputs[i] = torch.cat(
                    [
                        torch.zeros(shift, device=self.device),
                        fade(outputs[i, 0, shift:]),
                    ],
                    dim=-1,
                )
            else:
                fade = torchaudio.transforms.Fade(
                    fade_out_len=min(self.sample_rate, length - abs(shift))
                )
                outputs[i] = torch.cat(
                    [
                        fade(outputs[i, 0, :shift]),
                        torch.zeros(-shift, device=self.device),
                    ],
                    dim=-1,
                )

        return outputs


class Identity(Augmenter):
    def __init__(self, probability=1.0):
        super().__init__(probability)

    def forward(self, inputs):
        return inputs
