import os
import cv2
import shutil

import numpy as np
from tqdm import tqdm

from . import Sequence
from .const import imgs_dirname
from .interpolator import Interpolator
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str):
        assert os.path.isdir(input_dir), 'The input directory must exist'
        assert not os.path.exists(output_dir), 'The output directory must not exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir

        path = os.path.join(os.path.dirname(__file__), "../../pretrained_models/film_net/Style/saved_model")
        self.interpolator = Interpolator(path, None)

    def upsample(self):
        sequence_counter = 0
        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath)
            print(sequence)
            if sequence is None:
                continue
            sequence_counter += 1
            print('Processing sequence number {}'.format(src_absdirpath))
            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_dirname)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)
        timestamps_list = list()

        idx = 0
        for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
            I0 = img_pair[0][None]
            I1 = img_pair[1][None]
            t0, t1 = time_pair

            total_frames, total_timestamps = self._upsample_adaptive(I0, I1, t0, t1, int(input('请输入插帧倍数2^')))
            total_frames = [I0[0]] + total_frames
            timestamps = [t0] + total_timestamps

            sorted_indices = np.argsort(timestamps)
            total_frames = [total_frames[j] for j in sorted_indices]
            timestamps = [timestamps[i] for i in sorted_indices]

            timestamps_list += timestamps
            for frame in total_frames:
                self._write_npy(frame, idx, dest_imgs_dir)
                idx += 1

        timestamps_list.append(t1)
        self._write_npy(I1[0, ...], idx, dest_imgs_dir)
        self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections):
        if num_bisections == 0:
            return [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)


        left_images, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        return images, timestamps

    def _prepare_output_dir(self, src_dir: str, dest_dir: str):
        # Copy directory structure.
        def ignore_files(directory, files):
            return [f for f in files if os.path.isfile(os.path.join(directory, f))]
        shutil.copytree(src_dir, dest_dir, ignore=ignore_files)

    @staticmethod
    def _write_npy(img: np.ndarray, idx: int, imgs_dir: str):
        assert os.path.isdir(imgs_dir)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        path = os.path.join(imgs_dir, "%08d.npy" % idx)
        np.save(path, np.clip(img,0,1))


    @staticmethod
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        with open(timestamps_filename, 'w') as t_file:
            t_file.writelines([str(t) + '\n' for t in timestamps])