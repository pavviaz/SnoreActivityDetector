from typing import Any
from torchaudio.transforms import MelSpectrogram, MFCC


class BaseFeatureExtractor:
    def __init__(
            self,
            sample_rate=16000,
            win_length=160,
            hop_length=160,
            n_fft=256,
            n_mels=32,
            f_min=10,
            f_max=8000,
            power=1,
            pwr_to_db=True,
            top_db=80.0
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.win_length = win_length
        self.hop_length = hop_length

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        self.power = power
        self.pwr_to_db = pwr_to_db
        self.top_db = top_db


class MFCC_FE(BaseFeatureExtractor):
    def __init__(self, sample_rate=16000, win_length=160, hop_length=160, n_fft=256, n_mels=32, f_min=10, f_max=8000, power=1, pwr_to_db=True, top_db=80):
        super().__init__(sample_rate, win_length, hop_length, n_fft, n_mels, f_min, f_max, power, pwr_to_db, top_db)
        self.transform = MFCC(sample_rate=self.sample_rate,
                              n_mfcc=self.n_mels,
                              log_mels=True,
                              melkwargs= 
                              {
                                   "n_fft": self.n_fft, 
                                   "win_length": self.win_length,
                                   "hop_length": self.hop_length,
                                   "n_mels": self.n_mels, 
                                   "power": self.power,
                                   "f_min": self.f_min, 
                                   "f_max": self.f_max
                              })

    def __call__(self, audio, *args: Any, **kwds: Any) -> Any:
        return self.transform(audio)


class MelSpec_FE(BaseFeatureExtractor):
    def __init__(self, sample_rate=16000, win_length=160, hop_length=160, n_fft=256, n_mels=32, f_min=10, f_max=8000, power=1, pwr_to_db=True, top_db=80):
        super().__init__(sample_rate, win_length, hop_length, n_fft, n_mels, f_min, f_max, power, pwr_to_db, top_db)
        self.transform = MelSpectrogram(sample_rate=self.sample_rate,
                                        n_fft=self.n_fft, 
                                        win_length=self.win_length,
                                        hop_length=self.hop_length,
                                        n_mels=self.n_mels, 
                                        power=self.power,
                                        f_min=self.f_min, 
                                        f_max=self.f_max
                                        )

    def __call__(self, audio, *args: Any, **kwds: Any) -> Any:
        return self.transform(audio)


class Raw_FE(BaseFeatureExtractor):
    def __call__(self, audio, *args: Any, **kwds: Any) -> Any:
        return audio


if __name__ == "__main__":
    import torch
    audio = torch.randn((1, 48000))
    melspec = MelSpec_FE(win_length=160, n_fft=160, hop_length=480, n_mels=40)
    a = melspec(audio)

    mfcc = MFCC_FE(win_length=160, n_fft=160, hop_length=480, n_mels=40)
    b = melspec(audio)
    print()