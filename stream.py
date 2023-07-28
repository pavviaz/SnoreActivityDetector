import statistics
import torch

import yaml
import pyaudio
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
from model_loader import ModelLoader
from utils.checkpointing import Checkpointing
from feature_extractors import MelSpec_FE, MFCC_FE, Raw_FE


class StreamPrediction:
    """
    Class for predicting streaming data. Heavily adapted from the implementation:
    """
    def __init__(self, cfg_path):
        # with open(cfg_path, 'r') as f:
        #     self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_manager = ModelLoader(r"inference_models\CB_mfcc_clear")
        self.model = self.model_manager.model
        self.model.eval()

        # Recording parameters
        # self.sr = self.cfg['data']['sample_rate']
        self.sr = 16000
        self.chunk_duration = 2
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.window_duration = 2
        self.window_samples = int(self.sr * self.window_duration)
        self.silence_threshold = 100

        # self.fe = MelSpec_FE(sample_rate=self.sr,
        #                      win_length=160,
        #                      n_fft=160,
        #                      hop_length=480,
        #                      n_mels=40)
        
        self.fe = MFCC_FE(sample_rate=self.sr,
                             win_length=160,
                             n_fft=160,
                             hop_length=480,
                             n_mels=40)
        
        # self.fe = Raw_FE()

        # Data structures and buffers
        self.queue = Queue()
        # self.data = np.zeros(self.window_samples, dtype="float32")
        self.data = np.array([], dtype="float32")

        # Plotting parameters
        self.kw_target = 1
        self.change_bkg_frames = 2
        self.change_bkg_counter = 0
        self.change_bkg = False

    def start_stream(self):
        """
        Start audio data streaming from microphone
        :return: None
        """
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
            input_device_index=0,
            stream_callback=self.callback,
        )

        stream.start_stream()
        
        collect_to = int(self.window_samples / self.chunk_samples)
        result_buffer = [{"res": 1} for _ in range(collect_to)]
        try:
            while True:
                data = self.queue.get()
                
                with torch.no_grad():
                    data = torch.tensor(np.expand_dims(data, 0))
                    pred = self.model(torch.unsqueeze(self.fe(data), 0))

                if torch.argmax(pred, dim=-1).item() == self.kw_target:
                    print(f"KeyWord!; {pred.numpy()}", sep="", end="", flush=True)

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
            self.data = self.data[-self.window_samples:]
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
        plt.plot(data[-len(data) // 2:])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylabel("Amplitude")

        # Filterbank energies
        plt.subplot(312)
        plt.imshow(fbank[-fbank.shape[0] // 2:, :].T, aspect="auto")
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
    audio_stream = StreamPrediction("./data/config.yaml")
    audio_stream.start_stream()
