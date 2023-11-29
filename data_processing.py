import torch
import torchaudio
from ml_pipeline.meta_dicts import FEATURES
from feature_extractors.main_fe import MelSpec_FE
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
import pydub
import numpy as np


BETTER_MEL_CONF = {"n_fft": 256, "n_mels": 32, "power": 1}
BETTER_MEL_NG_CONF_POW = {"n_fft": 256, "n_mels": 32, "power": 2, "f_min": 10, "f_max": 8000}
BETTER_MEL_NG_CONF_AMP = {"n_fft": 256, "n_mels": 32, "power": 1, "f_min": 50, "f_max": 6000}

FEAT_CONF = {"better_mel": BETTER_MEL_CONF,
             "better_mel_ng_pow": BETTER_MEL_NG_CONF_POW,
             "better_mel_ng_amp": BETTER_MEL_NG_CONF_AMP }


class AudioDataset(Dataset):
    def __init__(self, audio, data_cfg, stride):
        self.mels = get_audio(audio, stride, data_cfg)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx][0], self.mels[idx][1]


#  https://stackoverflow.com/a/66922265
def pydub_to_pt(audio: pydub.AudioSegment):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return torch.tensor(np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1))).permute(-1, -2), audio.frame_rate


def ms2samples(ms, sample_rate):
    return int(ms * sample_rate / 1000)


def get_audio(audio, data_cfg, stride=64):
    # Loading audio from path or pydub segment
    if type(audio) == str:
        audio, sample_rate = torchaudio.load(audio, normalize=True)
        audio = audio.type(torch.float32)
    elif type(audio) == pydub.AudioSegment:     
        audio, sample_rate = pydub_to_pt(audio)
    else:
        raise TypeError
    
    audio = torch.mean(audio, dim=0, keepdim=True) if audio.shape[0] > 1 else audio
    
    if sample_rate != data_cfg.general.sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, data_cfg.general.sample_rate)
        audio = transform(audio)
        sample_rate = data_cfg.general.sample_rate
    
    # fe = FEATURES[data_cfg.general.feature_type]["cls"](
    #     sample_rate=sample_rate,
    #     win_length=ms2samples(data_cfg.general.win_len, sample_rate),
    #     n_fft=data_cfg.general.n_fft,
    #     hop_length=ms2samples(data_cfg.general.hop_len, sample_rate),
    #     n_mels=data_cfg.general.n_mels,
    # )
    
    audio = audio.permute(-1, -2)

    mel_audio = []

    samples_per_square = ms2samples(data_cfg.general.chunk_size, sample_rate)
    stride_samples = ms2samples(stride, sample_rate)

    # window loop
    for idx in range(len(audio)):
        l_bound = idx*stride_samples
        r_bound = l_bound+samples_per_square
        if r_bound > len(audio):
          break
        mel = audio[l_bound:r_bound, :].permute(-1, -2)
        mel_audio.append((mel, idx))

    return mel_audio


def dataset_from_audio(audio, data_cfg, stride=64, batch_size=32):
    """Genetrates dataset with mel-spectrograms extracted from specific audiofile or pydub segment

    Args:
        audio (Union[str, pydub.AudioSegment]): path to audiofile ; pydub segment
        mel_size (int): size of each mel
        stride (int, optional): controls frame step during mel-spectrograms extraction. Defaults to 64.
        batch_size (int, optional): Defaults to 32.

    Returns:
        tuple[DataLoader, int]: dataset objects and overall mels count
    """
    dataloader = AudioDataset(audio, stride, data_cfg)
    audio_dataset = DataLoader(dataloader, 
                               batch_size=batch_size, 
                               shuffle=False, 
                               drop_last=False)
    return audio_dataset, len(dataloader)