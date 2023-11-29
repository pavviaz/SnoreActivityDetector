import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from pydub import AudioSegment

from data_processing import dataset_from_audio


class SnoreInference:
    def __init__(
        self,
        ckpt_path: str,
        sample_rate: int,
        chunk_size: int,
        mean_overlaps: bool,
        threshold: int,
        stride: int,
        generate_clear_audio: str = None,
        include_probs: bool = False,
        label_path: str = "output/",
        clear_audio_path: str = "clear_audio/",
        batch_size: int = 128,
        device: str = "cpu",
    ):
        """
        Initializes the SnoreInference class
        object with the provided parameters.

        Args:
            ckpt_path (str): The path to the
            checkpoint file of the trained model.
            sample_rate (int): The sample rate of the audio.
            chunk_size (int): The size of each audio chunk for processing.
            mean_overlaps (bool): Flag indicating whether
            to calculate mean overlaps.
            threshold (int): The threshold value for labeling.
            stride (int): The stride value for segmenting the audio.
            generate_clear_audio (str, optional): The path to generate
            clear audio. Default is None.
            include_probs (bool, optional): Flag indicating whether to
            include probabilities in the labels. Default is False.
            label_path (str, optional): The path to save the output labels.
            Default is "output/".
            clear_audio_path (str, optional): The path to save the
            generated clear audio. Default is "clear_audio/".
            batch_size (int, optional): The batch size for inference.
            Default is 128.
            device (str, optional): The device to use for inference.
            Default is "cpu".

        Returns:
            None. The method initializes the class object.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.mean_overlaps = mean_overlaps
        self.threshold = threshold
        self.stride = stride
        self.generate_clear_audio = generate_clear_audio
        self.include_probs = include_probs
        self.ckpt_path = ckpt_path
        self.label_path = label_path
        self.clear_audio_path = clear_audio_path
        self.batch_size = batch_size
        self.device = device

        self.model = torch.jit.load(self.ckpt_path)
        self.model.to(self.device)

    def cut_empty_audio(self, audio_path, output_path, predicted_VA):
        """
        Generates a clean audio file by removing empty segments
        from the original audio file based on the predicted snore activity.

        Args:
            audio_path (str): The path to the original audio file
            output_path (str): The path to save the generated clean audio file
            predicted_VA (dict): A dictionary containing the
            predicted snore activity segments

        Returns:
            None. The method generates a clean audio file based
            on the predicted snore activity segments
            and saves it to the specified output path.
        """
        original_audio = AudioSegment.from_file(audio_path, format="wav")
        original_audio = original_audio.set_frame_rate(self.sample_rate)

        clean_audio = AudioSegment.empty()
        stride_segments = defaultdict(list)

        print(f"Generating clear audio to {output_path}...")

        [
            [
                stride_segments[(stride_segment - self.stride), stride_segment].append(
                    predicted_VA[p]
                )
                for p in predicted_VA.keys()
                if p[0] < stride_segment <= p[-1]
            ]
            for stride_segment in range(
                self.stride, (1 + len(predicted_VA)) * self.stride, self.stride
            )
        ]
        stride_segments = {
            k: round(np.mean(np.array(v).transpose(), axis=-1))
            for k, v in stride_segments.items()
        }

        for k, v in stride_segments.items():
            if v == 1:
                clean_audio += original_audio[k[0] : k[1]]

        clean_audio.export(f"{output_path}.wav", format="wav")
        print("Audio has been successfully generated")

    def labeling_func(self, x):
        return 1 if x[-1] > self.threshold else 0

    def predict(self, audio: str, output_file_name: str = None):
        """
        Predicts snore activity labels for
        an audio file and saves the labels in a JSON file.

        Args:
            audio (str): The path to the audio file for prediction.
            output_file_name (str, optional): The name of the output
            JSON file to save the labels. If not provided, a default
            name will be generated based on the audio
            file name, stride, and threshold.

        Returns:
            None
        """
        os.makedirs(self.label_path, exist_ok=True)
        audio_name = os.path.splitext(os.path.basename(audio))[0]
        output_file_name = os.path.join(
            self.label_path,
            f"{audio_name}_output_{self.stride}_stride_{self.threshold}_thr.json"
            if not output_file_name
            else output_file_name,
        )

        print(
            f"Processing {audio} with following config:\n \
                    output_path = {output_file_name}\n \
                    stride = {self.stride}\n \
                    threshold = {self.threshold}\n \
                    mean_overlap = {self.mean_overlaps}\n \
                    mel_size = {self.chunk_size}\n \
                    batch_size = {self.batch_size}"
        )

        labels = {}
        audio_dataset, segm_cnt = dataset_from_audio(
            audio,
            self.sample_rate,
            self.chunk_size,
            stride=self.stride,
            batch_size=self.batch_size,
        )
        print(f"{segm_cnt} segments were successfully extracted from audio")

        print(f"Starting inference...")
        t = time.time()
        for batch, (x_batch_val, y) in enumerate(audio_dataset):
            print(f"Processing batch {batch + 1} from {len(audio_dataset)}...")

            with torch.no_grad():
                output = torch.softmax(self.model(x_batch_val.to(self.device)), dim=-1)
            labels.update(
                {
                    (
                        k * self.stride,
                        k * self.stride + self.chunk_size,
                    ): v
                    for k, v in zip(
                        y.cpu().numpy().tolist(), output.cpu().numpy().tolist()
                    )
                }
            )
        torch.cuda.empty_cache()
        print(f"Inference completed. Total time = {time.time() - t} seconds")

        if self.mean_overlaps:
            mean_segments = [
                [labels[p] for p in labels.keys() if p[0] < stride_segment <= p[-1]]
                for stride_segment in range(
                    self.stride, (1 + len(labels)) * self.stride, self.stride
                )
            ]
            mean_segments = [
                np.mean(np.array(segm).transpose(), axis=-1) for segm in mean_segments
            ]

        labels = {
            k: [v, self.labeling_func(v)]
            if self.include_probs
            else self.labeling_func(v)
            for k, v in (
                labels.items()
                if not self.mean_overlaps
                else zip(labels.keys(), mean_segments)
            )
        }

        with open(output_file_name, "w+") as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=4)

        if self.generate_clear_audio:
            os.makedirs(self.clear_audio_path, exist_ok=True)
            self.cut_empty_audio(
                audio,
                os.path.join(
                    self.clear_audio_path,
                    f"clear_{audio_name}_{self.generate_clear_audio}",
                ),
                labels
                if not self.include_probs
                else {k: v[-1] for k, v in labels.items()},
            )

        print("JSON labels file has been successfully created")


if __name__ == "__main__":
    vad = SnoreInference(
        ckpt_path="model_weights/weights.pt",
        sample_rate=16000,
        chunk_size=1000,
        mean_overlaps=False,
        generate_clear_audio="jit_model",
        include_probs=True,
        threshold=0.25,
        stride=1000,
        device="cuda",
    )

    vad.predict("test_mic_audios/230810_0002_conv.wav")
