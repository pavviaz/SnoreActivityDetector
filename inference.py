import json
from typing import Union
import yaml
import numpy as np
import torch
import os
from ml_pipeline.model_manager import ModelManager
import pydub
import time
from collections import defaultdict
from pydub import AudioSegment
from data_proc_pt import dataset_from_audio


MEL_SIZE = 2000  # ms
THRESHOLDS = {0: 0.22, 1: 0.25, 2: 0.27, 3: 0.3}
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class SnoreInference:
    def __init__(
        self,
        cfg_path: str = None,
        mean_overlaps: bool = False,
        aggressiveness_mode: int = 1,
        exact_threshold_value: int = None,
        stride: int = 64,
        verbose: bool = False,
        generate_clear_audio: str = None,
        vad_ckpt_path: str = "models/torch_model_state_dict.pt",
        vad_log_path: str = "log/",
        vad_output_path: str = "output/",
        clear_audio_path: str = "clear_audio/",
        batch_size: int = 128,
        device: str = "cpu",
    ):
        """Voice Activity Detection (Pytorch) interface class.
        If config path is not None, config values override default ones.

        Args:
            cfg_path (str, optional): Path to the config. Defaults to None.

            mean_overlaps (bool, optional): if True, predictions will be meaned for overlapping
            segments (significantly improves accuracy). Defaults to False.

            aggressiveness_mode (int, optional): agressivennes value, one of (0, 1, 2, 3), which
            is equivalent to THRESHOLDS values respectively. Defaults to 1.

            exact_threshold_value (float, optional): This value will be used as threshold instead
            'aggressiveness_mode', if defined". Defaults to None.

            stride (int, optional): one of (64, 128, 192, 256, 320), controls frame step during
            mel-spectrograms extraction. Defaults to 64.

            verbose (bool, optional): if True, inference steps logging into file and stdout. Defaults to False.

            generate_clear_audio (str, optional): if specified, clear audio (voice only) will be generated with this name. Defaults to None.

            vad_ckpt_path (str, optional): Path to the VAD model. Defaults to "models/torch_model_state_dict.pt".

            vad_log_path (str, optional): Path to the logging folder. Defaults to "log/".

            vad_output_path (str, optional): Path to the output files folder. Defaults to "output/".

            clear_audio_path (str, optional): Path to the generated clear audio. Defaults to "clear_audio/".

            batch_size (int, optional): Defaults to 128.

            device (str, optional): Defaults to "cpu".
        """
        self.cfg_path = cfg_path
        self.mean_overlaps = mean_overlaps
        self.aggressiveness_mode = aggressiveness_mode
        self.exact_threshold_value = exact_threshold_value
        self.stride = stride
        self.verbose = verbose
        self.generate_clear_audio = generate_clear_audio
        self.vad_ckpt_path = vad_ckpt_path
        self.vad_log_path = vad_log_path
        self.vad_output_path = vad_output_path
        self.clear_audio_path = clear_audio_path
        self.batch_size = batch_size
        self.device = device

        if self.cfg_path and os.path.exists(self.cfg_path):
            with open(self.cfg_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)["vad_config"]
                self.__dict__.update(self.config)

        if (
            self.exact_threshold_value
            and THRESHOLD_MIN <= self.exact_threshold_value <= THRESHOLD_MAX
        ):
            self.threshold = self.exact_threshold_value
        else:
            self.aggressiveness_mode = (
                1
                if self.aggressiveness_mode not in list(THRESHOLDS.keys())
                else self.aggressiveness_mode
            )
            self.threshold = THRESHOLDS[self.aggressiveness_mode]

        if self.verbose:
            os.makedirs(self.vad_log_path, exist_ok=True)

        self.manager = ModelManager(model_name=self.vad_ckpt_path)
        self.model = self.manager.get_featured_model()
        self.model.to(self.device)

        self.data_cfg = self.manager.get_data_cfg()

    def cut_empty_audio(self, audio_path, output_path, predicted_VA):
        original_audio = AudioSegment.from_file(audio_path, format="wav")
        # original_audio = original_audio.set_frame_rate(SAMPLE_RATE)

        clean_audio = AudioSegment.empty()
        not_clean_audio = AudioSegment.empty()
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
            # else:
            #     not_clean_audio += original_audio[k[0]: k[1]]

        clean_audio.export(f"{output_path}.wav", format="wav")
        # not_clean_audio.export(f"{output_path}_dirty.wav", format="wav")
        print("Audio has been successfully generated")

    def labeling_func(self):
        return lambda x: 1 if x[-1] > self.threshold else 0
        # if not self.model.model_type == "M5":
        #     return lambda x: 1 if x[-1] > self.threshold else 0
        # return lambda x: int(np.argmax(x))

    def predict(
        self, audio: Union[str, pydub.AudioSegment], output_file_name: str = None
    ):
        """VAD inference function.
        Creates VAD label file to specific audio,
        or returns only prediction results (if audio is pydub segment).

        Args:
            audio (Union[str, pydub.AudioSegment]): path to specific audiofile ; pydub segment

            output_file_name (str, optional): label file path.
            Defaults to ``'{audio_name}_labels_{stride}_{threshold}_thr.json'``. Not used if audio is pydub.AudioSegment

        Raises:
            Exception: stride should be in ``0-{MEL_SIZE}`` bounds if not ``0 < stride <= MEL_SIZE``
            Exception: stride should be multiple of 64 if not ``stride % 64 == 0``
        """
        audio_is_path = type(audio) == str

        # if not 0 < self.stride <= MEL_SIZE:
        #     raise Exception(f"stride should be in 0-{MEL_SIZE} bounds")
        # if not self.stride % 64 == 0:
        #     raise Exception(f"stride should be multiple of 64")

        if audio_is_path:
            os.makedirs(self.vad_output_path, exist_ok=True)
            audio_name = os.path.splitext(os.path.basename(audio))[0]
            output_file_name = os.path.join(
                self.vad_output_path,
                f"{audio_name}_output_{self.stride}_stride_{self.threshold}_thr.json"
                if not output_file_name
                else output_file_name,
            )
        else:
            audio_name = None
            output_file_name = None

        print(
            f"Processing {audio} with following config:\n \
                    output_path = {output_file_name}\n \
                    stride = {self.stride}\n \
                    self.threshold = {self.threshold}\n \
                    mean_overlap={self.mean_overlaps}\n \
                    mel_size = {self.data_cfg.general.chunk_size}\n \
                    batch_size = {self.batch_size}"
        )

        labels = {}
        audio_dataset, segm_cnt = dataset_from_audio(
            audio,
            self.data_cfg,
            stride=self.stride,
            batch_size=self.batch_size,
        )
        print(f"{segm_cnt} segments were successfully extracted from audio")

        print(f"Starting inference...")
        t = time.time()
        for batch, (x_batch_val, y) in enumerate(audio_dataset):
            print(
                f"Processing batch {batch + 1} from {len(audio_dataset)}..."
            )

            with torch.no_grad():
                output = torch.softmax(
                    self.model(x_batch_val.to(self.device)), dim=-1
                )
            labels.update(
                {
                    (k * self.stride, k * self.stride + self.data_cfg.general.chunk_size): v
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

        if audio_is_path:
            # labels = {
            #     k: [v, self.labeling_func()(v)]
            #     for k, v in (
            #         labels.items()
            #         if not self.mean_overlaps
            #         else zip(labels.keys(), mean_segments)
            #     )
            # }
            labels = {k: self.labeling_func()(v) for k, v in
                      (labels.items() if not self.mean_overlaps else zip(labels.keys(), mean_segments))}
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
                    labels,
                )

            print("JSON labels file has been successfully created")
            return labels
        else:
            ret_list = list(labels.values())
            return ret_list[0] if len(ret_list) else []


if __name__ == "__main__":
    vad = SnoreInference(
        mean_overlaps=False,
        verbose=True,
        vad_ckpt_path="SAD_M5E_kaggle_test",
        generate_clear_audio="SAD_M5E_kaggle_test",
        exact_threshold_value=0.4,
        stride=1000,
        device="cpu",
    )

    vad.predict("test_mic_audios/230810_0002_conv.wav")
    print("Finish")