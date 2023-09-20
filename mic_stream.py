import torch
from queue import Queue

import pydub
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

from ml_pipeline.model_manager import ModelManager
from ml_pipeline.meta_dicts import FEATURES


def eq(inp, target_dbfs):
    inputs_ndim = len(inp.shape)
    rms = torch.sqrt(torch.mean(inp ** 2, dim=inputs_ndim - 1, keepdim=True))
    rms = torch.clamp(rms, 1e-10, 1e10)

    wave_dBFS = 20 * torch.log10(rms)
    dBFS_diff = target_dbfs - wave_dBFS

    return inp * (10 ** (dBFS_diff / 20))


class StreamPrediction:
    """
    Class for predicting streaming data. Heavily adapted from the implementation:
    """

    def __init__(self, model_name):
        manager = ModelManager(model_name=model_name)

        self.model = manager.get_model()
        # self.model = manager.get_featured_model()
        self.model.to("cpu")
        self.model.eval()

        data_cfg = manager.get_data_cfg()
        self.sr = data_cfg.general.sample_rate
        self.chunk_duration = data_cfg.general.mel_size / 1000
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.window_duration = data_cfg.general.mel_size / 1000
        self.window_samples = int(self.sr * self.window_duration)
        self.silence_threshold = 100

        self.classes = data_cfg.general.training_tokens

        self.fe = FEATURES[data_cfg.general.feature_type]["cls"](
            sample_rate=self.sr,
            win_length=self.ms2samples(data_cfg.general.win_len, self.sr),
            n_fft=data_cfg.general.n_fft,
            hop_length=self.ms2samples(data_cfg.general.hop_len, self.sr),
            n_mels=data_cfg.general.n_mels,
        )

        # Data structures and buffers
        self.queue = Queue()
        self.data = np.zeros(self.window_samples, dtype="float32")

        # Plotting parameters
        self.kw_target = 2
        self.change_bkg_frames = 2
        self.change_bkg_counter = 0
        self.change_bkg = False

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def pydub_to_pt(self, audio: pydub.AudioSegment):
        """
        Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
        where each value is in range [-1.0, 1.0].
        Returns tuple (audio_np_array, sample_rate).
        """
        return torch.tensor(
            np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(
                (-1, audio.channels)
            )
            / (1 << (8 * audio.sample_width - 1))
        ).permute(-1, -2)

    def ms2samples(self, ms, sample_rate):
        return int(ms * sample_rate / 1000)

    def start_stream(self):
        """
        Start audio data streaming from microphone
        :return: None
        """
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
            input_device_index=1,
            stream_callback=self.callback,
        )

        stream.start_stream()

        try:
            while True:
                data = self.queue.get()
                data = torch.tensor(np.expand_dims(data, 0))

                with torch.no_grad():
                    pred = self.model((torch.unsqueeze(self.fe(eq(data, -20)), 0)))
                    # pred = self.model((torch.unsqueeze(data, 0)))

                print(
                    f"{torch.softmax(pred, -1)} {self.classes[torch.argmax(pred, dim=-1).item()]}; ",
                    sep="",
                    end="\n",
                    flush=True,
                )

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        """
        Obtain the data from buffer and load it to queue
        :param in_data: Daa buffer
        :param frame_count: Frame count
        :param time_info: Time information
        :param status: Status
        """
        data0 = np.frombuffer(in_data, dtype="float32")

        if np.abs(data0).mean() < self.silence_threshold:
            print(".", sep="", end="", flush=True)
        else:
            print("-", sep="", end="", flush=True)

        self.data = np.append(self.data, data0)

        if len(self.data) > self.window_samples:
            self.data = self.data[-self.window_samples :]
            self.queue.put(self.data)

        return in_data, pyaudio.paContinue

    def plotter(self, data, fbank, pred):
        """
        Plot waveform, filterbank energies and hotword presence
        :param data: Audio data array
        :param fbank: Log Mel filterbank energies
        :param pred: Prediction
        """
        plt.clf()

        # Wave
        plt.subplot(311)
        plt.plot(data[-len(data) // 2 :])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylabel("Amplitude")

        # Filterbank energies
        plt.subplot(312)
        plt.imshow(fbank[-fbank.shape[0] // 2 :, :].T, aspect="auto")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().invert_yaxis()
        plt.ylim(0, 40)
        plt.ylabel(r"$\log \, E_{m}$")

        # Hotword detection
        plt.subplot(313)
        ax = plt.gca()

        if pred == self.kw_target:
            self.change_bkg = True

        if self.change_bkg and self.change_bkg_counter < self.change_bkg_frames:
            ax.set_facecolor("lightgreen")

            ax.text(
                x=0.5,
                y=0.5,
                s="KeyWord!",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=30,
                color="red",
                fontweight="bold",
                transform=ax.transAxes,
            )

            self.change_bkg_counter += 1
        else:
            ax.set_facecolor("salmon")
            self.change_bkg = False
            self.change_bkg_counter = 0

        plt.tight_layout()
        plt.pause(0.01)


if __name__ == "__main__":
    audio_stream = StreamPrediction("M5E_greek_clear")
    audio_stream.start_stream()
