from queue import Queue
import argparse
from datetime import datetime
import os

import torch
import pyaudio
import numpy as np
import yaml
from munch import munchify


LOG_NAME = "log.txt"


class StreamPrediction:
    def __init__(self, cfg_path):
        with open(cfg_path, encoding="utf-8") as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.cfg = munchify(config)
        self.model = torch.jit.load(self.cfg.ckpt_path)

        self.chunk_samples = int(self.cfg.sample_rate * 500 / 1000)
        self.window_samples = int(self.cfg.sample_rate * 1000 / 1000)
        self.silence_threshold = 100

        self.queue = Queue()
        self.data = np.zeros(self.window_samples, dtype="float32")

    def start_stream(self):
        """
        Start audio data streaming from microphone
        :return: None
        """
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.cfg.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_samples,
            stream_callback=self.callback,
        )

        stream.start_stream()

        try:
            while True:
                data = self.queue.get()
                data = np.expand_dims(data, 0)
                data = np.expand_dims(data, 0)

                with torch.no_grad():
                    pred = self.model(torch.tensor(data))

                log_str = f"{datetime.now().strftime('%H:%M:%S:%f')} | {torch.softmax(pred, -1)};"

                with open(os.path.join(self.cfg.label_path, LOG_NAME), "a+") as f:
                    f.write(log_str + "\n")

                print(
                    log_str,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    audio_stream = StreamPrediction(args.config_path)
    audio_stream.start_stream()
