import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):
    def __init__(self, audio, sample_rate, chunk_size, stride):
        self.mels = get_audio(audio, stride, sample_rate, chunk_size)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx][0], self.mels[idx][1]


def ms2samples(ms, sample_rate):
    return int(ms * sample_rate / 1000)


def get_audio(audio, stride=64, sample_rate=16000, chunk_size=1000):
    """
    Takes an audio file as input and
    returns a list of chunks of the audio
    file, along with their corresponding indices.

    Args:
        audio (str): The path to the audio file.
        stride (int, optional): The stride (in milliseconds) of
        the sliding window. Default is 64.
        sample_rate (int, optional): The sample rate of the audio file.
        Default is 16000.
        chunk_size (int, optional): The duration (in milliseconds) of
        each chunk. Default is 1000.

    Returns:
        list: A list of tuples, where each tuple
        contains audio features of a chunk and its corresponding index.
    """
    audio, rsample_rate = torchaudio.load(audio, normalize=True)
    audio = audio.type(torch.float32)

    audio = torch.mean(audio, dim=0, keepdim=True) if audio.shape[0] > 1 else audio

    if rsample_rate != sample_rate:
        transform = torchaudio.transforms.Resample(rsample_rate, sample_rate)
        audio = transform(audio)
        rsample_rate = sample_rate

    audio = audio.permute(-1, -2)

    mel_audio = []

    samples_per_square = ms2samples(chunk_size, rsample_rate)
    stride_samples = ms2samples(stride, rsample_rate)

    # window loop
    for idx in range(len(audio)):
        l_bound = idx * stride_samples
        r_bound = l_bound + samples_per_square
        if r_bound > len(audio):
            break
        mel = audio[l_bound:r_bound, :].permute(-1, -2)
        mel_audio.append((mel, idx))

    return mel_audio


def dataset_from_audio(audio, sample_rate, chunk_size, stride=64, batch_size=32):
    """
    Create a PyTorch DataLoader object from an audio file.

    Args:
        audio (str): The path to the audio file.
        sample_rate (int): The sample rate of the audio file.
        chunk_size (int): The duration (in milliseconds) of each chunk.
        stride (int, optional): The stride (in milliseconds) of
        the sliding window. Default is 64.
        batch_size (int, optional): The batch size for the DataLoader.
        Default is 32.

    Returns:
        audio_dataset (DataLoader): A PyTorch DataLoader object
        containing the audio dataset.
        dataset_length (int): The length of the audio dataset.
    """
    dataloader = AudioDataset(audio, sample_rate, chunk_size, stride)
    audio_dataset = DataLoader(
        dataloader, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return audio_dataset, len(dataloader)
