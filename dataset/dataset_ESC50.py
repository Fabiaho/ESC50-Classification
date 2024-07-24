import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np
import wavio
import random

import config
from . import transforms as U

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    # import wget
    import zipfile

    root = os.path.dirname(file_path)
    # wget.download(url, out=file_path, bar=download_progress)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(root)


# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def resize_tensor(tensor, target_shape):
    # Resize tensor to the target shape by padding or cropping
    current_shape = tensor.shape
    if current_shape == target_shape:
        return tensor
    padded_tensor = torch.zeros(target_shape)
    min_shape = [
        min(current_shape[i], target_shape[i]) for i in range(len(target_shape))
    ]
    slices = tuple(slice(0, min_shape[i]) for i in range(len(target_shape)))
    padded_tensor[slices] = tensor[slices]
    return padded_tensor


def manual_pad(array, pad_width, constant_value=0):
    """
    Manually pad an array with a constant value.

    :param array: Input array to be padded
    :param pad_width: Width of padding
    :param constant_value: Value to pad with
    :return: Padded array
    """
    padded_shape = tuple(dim + 2 * pad_width for dim in array.shape)
    padded_array = np.full(padded_shape, constant_value, dtype=array.dtype)
    slices = tuple(slice(pad_width, pad_width + dim) for dim in array.shape)
    padded_array[slices] = array
    return padded_array


class ESC50(data.Dataset):

    def __init__(
        self, root, test_folds=frozenset((1,)), subset="train", download=False
    ):
        audio = "ESC-50-master/audio"
        root = os.path.normpath(root)
        audio = os.path.join(root, audio)
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError
        # path = path.split(os.sep)
        if not os.path.exists(audio) and download:
            os.makedirs(root, exist_ok=True)
            file_name = "master.zip"
            file_path = os.path.join(root, file_name)
            url = f"https://github.com/karoldvl/ESC-50/archive/{file_name}"
            download_extract_zip(url, file_path)

        self.root = audio
        # getting name of all files inside the all the train_folds
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split("-")[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - test_folds
        train_files = [f for f in temp if int(f.split("-")[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split("-")[0]) in test_folds]
        # sanity check
        assert set(temp) == (set(train_files) | set(test_files))
        if subset == "test":
            self.file_names = test_files
        else:
            """if config.val_size:
                train_files, val_files = train_test_split(
                    train_files, test_size=config.val_size, random_state=0
                )"""
            if subset == "train":
                self.file_names = train_files
            else:
                self.file_names = train_files#val_files

        self.esc50_sounds = []
        self.esc50_labels = []

        for file_name in self.file_names:
            sound = wavio.read(os.path.join(self.root, file_name)).data.T[0]
            start = sound.nonzero()[0].min()
            end = sound.nonzero()[0].max()
            sound = sound[start : end + 1]  # Remove silent sections
            label = int(os.path.splitext(file_name)[0].split("-")[-1])
            self.esc50_sounds.append(sound)
            self.esc50_labels.append(label)

        if self.subset == "train":
            self.preprocess = U.Compose(
                U.RandomGain(1.25),
                U.Padding(config.inputLength // 2),
                U.RandomCrop(config.inputLength),
                U.Normalize(32768.0),
            )

            self.augmentation = U.Compose(U.RandomGain(6))
        else:
            self.preprocess = U.Compose(
                U.Padding(config.inputLength // 2),
                U.Normalize(32768.0),
                U.MultiCrop(config.inputLength, 10),
            )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # file_name = self.file_names[index]

        if self.subset == "train":
            while True:
                rand1 = random.randint(0, len(self.esc50_sounds) - 1)
                rand2 = random.randint(0, len(self.esc50_sounds) - 1)

                wave1 = self.esc50_sounds[rand1]
                wave2 = self.esc50_sounds[rand2]

                target1 = self.esc50_labels[rand1]
                target2 = self.esc50_labels[rand2]

                if target1 != target2:
                    break

            # pad_width = config.inputLength // 2
            # wave1 = manual_pad(wave1, pad_width)
            # wave2 = manual_pad(wave2, pad_width)
            # print("Preprocessing...")
            wave1 = self.preprocess(wave1)
            wave2 = self.preprocess(wave2)

            # Mix two examples
            # print("Mixing...")
            r = np.array(random.random())
            sound = U.mix(wave1, wave2, r, config.sr).astype(np.float32)
            eye = np.eye(config.n_classes)
            label = (eye[target1] * r + eye[target2] * (1 - r)).astype(np.float32)

            # For stronger augmentation
            # print("Augmenting...")
            sound = self.augmentation(sound).astype(np.float32)
            # print("Done")
            file_name = "mix"
        else:
            sound = self.esc50_sounds[index]
            target = self.esc50_labels[index]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.zeros((10, config.n_classes))
            label[:, target] = 1
            file_name = self.file_names[index]

        # spec = torch.cat((log_s, delta, delta_delta), dim=0)

        return file_name, sound, label
