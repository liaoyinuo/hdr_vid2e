import os
import numpy as np
from pathlib import Path
from typing import Union


class Sequence:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageSequence(Sequence):
    def __init__(self, imgs_dirpath: str, fps: float):
        super().__init__()
        self.fps = fps

        assert os.path.isdir(imgs_dirpath)
        self.imgs_dirpath = imgs_dirpath

        self.file_names = [f for f in os.listdir(imgs_dirpath) if self._is_img_file(f)]
        assert self.file_names
        self.file_names.sort()

    @classmethod
    def _is_img_file(cls, path: str):
        return Path(path).suffix.lower() == '.npy'

    def __next__(self):
        for idx in range(0, len(self.file_names) - 1):
            file_paths = self._get_path_from_name([self.file_names[idx], self.file_names[idx + 1]])
            imgs = [np.load(f) for f in file_paths]
            times_sec = [idx/self.fps, (idx + 1)/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return len(self.file_names) - 1

    def _get_path_from_name(self, file_names: Union[list, str]) -> Union[list, str]:
        if isinstance(file_names, list):
            return [os.path.join(self.imgs_dirpath, f) for f in file_names]
        return os.path.join(self.imgs_dirpath, file_names)


