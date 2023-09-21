import os
from typing import Union

from .const import imgs_dirname
from .dataset import Sequence, ImageSequence



def get_imgs_directory(dirpath: str) -> Union[None, str]:
    imgs_dir = os.path.join(dirpath, imgs_dirname)
    if os.path.isdir(imgs_dir):
        return imgs_dir
    return None


def get_sequence_or_none(dirpath: str) -> Union[None, Sequence]:
    fps = int(input('请输入FPS: '))
    return ImageSequence(dirpath, fps)


