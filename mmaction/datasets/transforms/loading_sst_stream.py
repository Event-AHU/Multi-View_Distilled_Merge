import copy as cp
import os
import os.path as osp
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from mmaction.registry import TRANSFORMS
import random

@TRANSFORMS.register_module()
class RawFrameDecodeSSTWithStream(BaseTransform):
    

    def __init__(self,
                 sst_rgb_root: str,
                 stream_root: str,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 align_stream_with_rgb: bool = True,
                 file_client_args: Optional[dict] = None,
                 sst_filename_tmpl: Optional[str] = None,
                 sst_index_offset: int = 0,
                 img_suffixes: tuple = ('.png', '.jpg', '.jpeg')) -> None:
        self.sst_rgb_root = sst_rgb_root
        self.stream_root = stream_root
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.align_stream_with_rgb = align_stream_with_rgb

        self.file_client_args = file_client_args or {}
        self.file_client = None


        self.sst_filename_tmpl = sst_filename_tmpl
        self.sst_index_offset = int(sst_index_offset)
        self.img_suffixes = img_suffixes

    def _infer_sst_tmpl(self, directory_rgb: str) -> str:
        
        try:
            files = os.listdir(directory_rgb)
        except FileNotFoundError:
            raise FileNotFoundError(f'SST rgb directory not found: {directory_rgb}')

        cand = [f for f in files if osp.splitext(f)[1].lower() in self.img_suffixes]
        if len(cand) == 0:
            raise FileNotFoundError(f'No image files found in {directory_rgb}')

        cand.sort()
        sample = cand[0]
        stem, ext = osp.splitext(sample)


        if not stem.isdigit():

            return '{:08d}' + ext

        width = len(stem)
        return '{:0' + str(width) + 'd}' + ext

    def transform(self, results: dict) -> dict:
        mmcv.use_backend(self.decoding_backend)

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.file_client_args)


        directory_tsc = results['frame_dir']
        vid1, vid2 = directory_tsc.split('/')[-2], directory_tsc.split('/')[-1]
        directory_rgb = osp.join(self.sst_rgb_root, vid1, vid2)

        modality = results['modality']
        if modality != 'RGB':
            raise NotImplementedError(f'Only RGB supported, got {modality}')

        frame_inds = results['frame_inds']
        if frame_inds.ndim != 1:
            frame_inds = np.squeeze(frame_inds)


        start_index = int(results.get('start_index', 1))
        frame_inds_0 = frame_inds.astype(np.int64) - start_index
        frame_inds_0 = np.maximum(frame_inds_0, 0)

        offset = int(results.get('offset', 0))


        sst_tmpl = self.sst_filename_tmpl
        if sst_tmpl is None:
            sst_tmpl = self._infer_sst_tmpl(directory_rgb)

        imgs = []
        cache = {}
        for i, idx0 in enumerate(frame_inds_0):
            if idx0 in cache:
                imgs.append(cp.deepcopy(imgs[cache[idx0]]))
                continue
            cache[idx0] = i


            frame_id = int(idx0 + offset + self.sst_index_offset)
            filepath = osp.join(directory_rgb, sst_tmpl.format(frame_id))
            img_bytes = self.file_client.get(filepath)
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            imgs.append(cur_frame)


        stream_dir = osp.join(self.stream_root, vid1, vid2)
        fileLIST = os.listdir(stream_dir)
        fileLIST.sort()
        total_frames_stream = len(fileLIST)
        if total_frames_stream == 0:
            raise FileNotFoundError(f'No stream frames in {stream_dir}')

        if self.align_stream_with_rgb:
            inds = np.mod(frame_inds_0.astype(np.int64), total_frames_stream)
        else:
            start = random.randint(0, total_frames_stream - 1)
            inds = np.mod(start + np.arange(len(frame_inds_0)), total_frames_stream)

        stream_frame_all = []
        for idx in inds:
            data_path = osp.join(stream_dir, fileLIST[int(idx)])
            load_data = np.load(data_path, allow_pickle=True)
            stream_frame_all.append(load_data['stream_frames'])

        results['imgs'] = imgs
        results['stream_frame_all'] = stream_frame_all
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'sst_rgb_root={self.sst_rgb_root}, '
                f'stream_root={self.stream_root}, '
                f'io_backend={self.io_backend}, '
                f'decoding_backend={self.decoding_backend}, '
                f'align_stream_with_rgb={self.align_stream_with_rgb})')
