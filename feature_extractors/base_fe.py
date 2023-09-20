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
        top_db=80.0,
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
