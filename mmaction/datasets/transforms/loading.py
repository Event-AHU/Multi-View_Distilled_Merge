
import copy as cp
import io
import os
import os.path as osp
import shutil
from typing import Dict, List, Optional, Union

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from mmaction.registry import TRANSFORMS
from mmaction.utils import get_random_string, get_shm_dir, get_thread_id
import pdb
import random
@TRANSFORMS.register_module()
class LoadRGBFromFile(BaseTransform):
    

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 io_backend: str = 'disk',
                 ignore_empty: bool = False,
                 **kwargs) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client = FileClient(io_backend, **kwargs)
        self.io_backend = io_backend

    def transform(self, results: dict) -> dict:
        

        filename = results['img_path']
        try:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order='rgb',
                backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f"io_backend='{self.io_backend}')")
        return repr_str


@TRANSFORMS.register_module()
class LoadHVULabel(BaseTransform):
    

    def __init__(self, **kwargs):
        self.hvu_initialized = False
        self.kwargs = kwargs

    def init_hvu_info(self, categories, category_nums):
        
        assert len(categories) == len(category_nums)
        self.categories = categories
        self.category_nums = category_nums
        self.num_categories = len(self.categories)
        self.num_tags = sum(self.category_nums)
        self.category2num = dict(zip(categories, category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] + self.category_nums[i])
        self.category2startidx = dict(zip(categories, self.start_idx))
        self.hvu_initialized = True

    def transform(self, results):
        

        if not self.hvu_initialized:
            self.init_hvu_info(results['categories'], results['category_nums'])

        onehot = torch.zeros(self.num_tags)
        onehot_mask = torch.zeros(self.num_tags)
        category_mask = torch.zeros(self.num_categories)

        for category, tags in results['label'].items():

            if category not in self.categories:
                continue
            category_mask[self.categories.index(category)] = 1.
            start_idx = self.category2startidx[category]
            category_num = self.category2num[category]
            tags = [idx + start_idx for idx in tags]
            onehot[tags] = 1.
            onehot_mask[start_idx:category_num + start_idx] = 1.

        results['label'] = onehot
        results['mask'] = onehot_mask
        results['category_mask'] = category_mask
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hvu_initialized={self.hvu_initialized})')
        return repr_str


@TRANSFORMS.register_module()
class SampleFrames(BaseTransform):
    

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int,
                         ori_clip_len: float) -> np.array:
        

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        
        if self.clip_len == 1:

            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1

                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        
        if self.target_fps is not None:

            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        
        total_frames = results['total_frames']


        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class UniformSample(BaseTransform):
    

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False) -> None:

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def _get_sample_clips(self, num_frames: int) -> np.array:
        
        seg_size = float(num_frames - 1) / self.clip_len
        inds = []
        if not self.test_mode:
            for i in range(self.clip_len):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                inds.append(np.random.randint(start, end + 1))
        else:
            duration = seg_size / (self.num_clips + 1)
            for k in range(self.num_clips):
                for i in range(self.clip_len):
                    start = int(np.round(seg_size * i))
                    frame_index = start + int(duration * (k + 1))
                    inds.append(frame_index)

        return np.array(inds)

    def transform(self, results: dict):
        num_frames = results['total_frames']

        inds = self._get_sample_clips(num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}')
        return repr_str


@TRANSFORMS.register_module()
class UntrimmedSampleFrames(BaseTransform):
    

    def __init__(self, clip_len=1, frame_interval=16):
        self.clip_len = clip_len
        self.frame_interval = frame_interval

    def transform(self, results):
        
        total_frames = results['total_frames']
        start_index = results['start_index']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]

        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval})')
        return repr_str


@TRANSFORMS.register_module()
class DenseSampleFrames(SampleFrames):
    

    def __init__(self,
                 *args,
                 sample_range: int = 64,
                 num_sample_positions: int = 10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames: int) -> np.array:
        
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames: int) -> np.array:
        
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets

    def _sample_clips(self, num_frames: int) -> np.array:
        
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def transform(self, results: dict) -> dict:
        
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'sample_range={self.sample_range}, '
                    f'num_sample_positions={self.num_sample_positions}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        if not self.test_mode:
            frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

    def transform(self, results):
        
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)
        start_index = results.get('start_index', 0)

        frame_inds = np.array(frame_inds, dtype=np.int32) + start_index
        results['frame_inds'] = frame_inds
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class PyAVInit(BaseTransform):
    

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results):
        
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend})'
        return repr_str


@TRANSFORMS.register_module()
class PyAVDecode(BaseTransform):
    

    def __init__(self, multi_thread=False, mode='accurate'):
        self.multi_thread = multi_thread
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    @staticmethod
    def frame_generator(container, stream):
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def transform(self, results):
        
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        if self.mode == 'accurate':

            max_inds = max(results['frame_inds'])
            i = 0
            for frame in container.decode(video=0):
                if i > max_inds + 1:
                    break
                imgs.append(frame.to_rgb().to_ndarray())
                i += 1



            results['imgs'] = [
                imgs[i % len(imgs)] for i in results['frame_inds']
            ]
        elif self.mode == 'efficient':
            for frame in container.decode(video=0):
                backup_frame = frame
                break
            stream = container.streams.video[0]
            for idx in results['frame_inds']:
                pts_scale = stream.average_rate * stream.time_base
                frame_pts = int(idx / pts_scale)
                container.seek(
                    frame_pts, any_frame=False, backward=True, stream=stream)
                frame = self.frame_generator(container, stream)
                if frame is not None:
                    imgs.append(frame)
                    backup_frame = frame
                else:
                    imgs.append(backup_frame)
            results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['video_reader'] = None
        del container

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread}, mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class PIMSInit(BaseTransform):
    

    def __init__(self, io_backend='disk', mode='accurate', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def transform(self, results):
        
        try:
            import pims
        except ImportError:
            raise ImportError('Please run "conda install pims -c conda-forge" '
                              'or "pip install pims" to install pims first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        if self.mode == 'accurate':
            container = pims.PyAVReaderIndexed(file_obj)
        else:
            container = pims.PyAVReaderTimed(file_obj)

        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(io_backend={self.io_backend}, '
                    f'mode={self.mode})')
        return repr_str


@TRANSFORMS.register_module()
class PIMSDecode(BaseTransform):
    

    def transform(self, results):
        

        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = [container[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@TRANSFORMS.register_module()
class PyAVDecodeMotionVector(PyAVDecode):
    

    @staticmethod
    def _parse_vectors(mv, vectors, height, width):
        
        (w, h, src_x, src_y, dst_x,
         dst_y) = (vectors['w'], vectors['h'], vectors['src_x'],
                   vectors['src_y'], vectors['dst_x'], vectors['dst_y'])
        val_x = dst_x - src_x
        val_y = dst_y - src_y
        start_x = dst_x - w // 2
        start_y = dst_y - h // 2
        end_x = start_x + w
        end_y = start_y + h
        for sx, ex, sy, ey, vx, vy in zip(start_x, end_x, start_y, end_y,
                                          val_x, val_y):
            if (sx >= 0 and ex < width and sy >= 0 and ey < height):
                mv[sy:ey, sx:ex] = (vx, vy)

        return mv

    def transform(self, results):
        
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])


        max_idx = max(results['frame_inds'])
        i = 0
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec_context.options = {'flags2': '+export_mvs'}
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i > max_idx + 1:
                    break
                i += 1
                height = frame.height
                width = frame.width
                mv = np.zeros((height, width, 2), dtype=np.int8)
                vectors = frame.side_data.get('MOTION_VECTORS')
                if frame.key_frame:

                    assert vectors is None
                if vectors is not None and len(vectors) > 0:
                    mv = self._parse_vectors(mv, vectors.to_ndarray(), height,
                                             width)
                imgs.append(mv)

        results['video_reader'] = None
        del container



        results['motion_vectors'] = np.array(
            [imgs[i % len(imgs)] for i in results['frame_inds']])
        return results


@TRANSFORMS.register_module()
class DecordInit(BaseTransform):
    

    def __init__(self,
                 io_backend: str = 'disk',
                 num_threads: int = 1,
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def _get_video_reader(self, filename: str) -> object:
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        return container

    def transform(self, results: Dict) -> Dict:
        
        container = self._get_video_reader(results['filename'])
        results['total_frames'] = len(container)

        results['video_reader'] = container
        results['avg_fps'] = container.get_avg_fps()
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@TRANSFORMS.register_module()
class DecordDecode(BaseTransform):
    

    def __init__(self, mode: str = 'accurate') -> None:
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container: object,
                            frame_inds: np.ndarray) -> List[np.ndarray]:
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':

            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def transform(self, results: Dict) -> Dict:
        
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class OpenCVInit(BaseTransform):
    

    def __init__(self, io_backend: str = 'disk', **kwargs) -> None:
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != 'disk':
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(),
                                       f'{random_string}_{thread_id}')
            os.mkdir(self.tmp_folder)

    def transform(self, results: dict) -> dict:
        
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()

            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        if self.tmp_folder and osp.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend})')
        return repr_str


@TRANSFORMS.register_module()
class OpenCVDecode(BaseTransform):
    

    def transform(self, results: dict) -> dict:
        
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]

            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)

        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@TRANSFORMS.register_module()
class RawFrameDecode(BaseTransform):
    

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']


        directory_aps = directory + '/' + directory[-26:] + '_aps'
        directory_dvs = directory + '/' + directory[-26:] +'_dvs'
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        imgs_event = list()






        results['frame_inds'] = results['frame_inds']-1


        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        cache = {}
        results_0 = results['frame_inds'] - 0
        for i, frame_idx in enumerate(results_0):

            if frame_idx in cache:
                if modality == 'RGB':
                    imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                    imgs_event.append(cp.deepcopy(imgs_event[cache[frame_idx]]))
                else:
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx]]))
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx] + 1]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory_aps, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)

                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)


                filepath_event = osp.join(directory_dvs, filename_tmpl.format(frame_idx))
                img_event_bytes = self.file_client.get(filepath_event)
                cur_frame_event = mmcv.imfrombytes(img_event_bytes, channel_order='rgb')
                imgs_event.append(cur_frame_event)

            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError


        list_img = []

        list_img.extend(imgs)
        list_img.extend(imgs_event)


        results['imgs'] = list_img

        results['imgs_event'] = imgs_event
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]


        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerPackInput(BaseTransform):

    def __init__(self,
                 input_format='video',
                 filename_tmpl='img_{:05}.jpg',
                 modality='RGB',
                 start_index=1) -> None:
        self.input_format = input_format
        self.filename_tmpl = filename_tmpl
        self.modality = modality
        self.start_index = start_index

    def transform(self, video: Union[str, np.ndarray, dict]) -> dict:
        if self.input_format == 'dict':
            results = video
        elif self.input_format == 'video':
            results = dict(
                filename=video, label=-1, start_index=0, modality='RGB')
        elif self.input_format == 'rawframes':
            import re





            pattern = f'^{self.filename_tmpl}$'
            if self.modality == 'Flow':
                pattern = pattern.replace('{}', 'x')
            pattern = pattern.replace(
                pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')
            total_frames = len(
                list(
                    filter(lambda x: re.match(pattern, x) is not None,
                           os.listdir(video))))
            results = dict(
                frame_dir=video,
                total_frames=total_frames,
                label=-1,
                start_index=self.start_index,
                filename_tmpl=self.filename_tmpl,
                modality=self.modality)
        elif self.input_format == 'array':
            modality_map = {2: 'Flow', 3: 'RGB'}
            modality = modality_map.get(video.shape[-1])
            results = dict(
                total_frames=video.shape[0],
                label=-1,
                start_index=0,
                array=video,
                modality=modality)

        return results


@TRANSFORMS.register_module()
class ArrayDecode(BaseTransform):
    

    def transform(self, results):
        

        modality = results['modality']
        array = results['array']

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for i, frame_idx in enumerate(results['frame_inds']):

            frame_idx += offset
            if modality == 'RGB':
                imgs.append(array[frame_idx])
            elif modality == 'Flow':
                imgs.extend(
                    [array[frame_idx, ..., 0], array[frame_idx, ..., 1]])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class ImageDecode(BaseTransform):
    

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results):
        
        mmcv.use_backend(self.decoding_backend)

        filename = results['filename']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class AudioDecodeInit(BaseTransform):
    

    def __init__(self,
                 io_backend: str = 'disk',
                 sample_rate: int = 16000,
                 pad_method: str = 'zero',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.sample_rate = sample_rate
        if pad_method in ['random', 'zero']:
            self.pad_method = pad_method
        else:
            raise NotImplementedError
        self.kwargs = kwargs
        self.file_client = None

    @staticmethod
    def _zero_pad(shape: int) -> np.ndarray:
        
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape: int) -> np.ndarray:
        

        return np.random.rand(shape).astype(np.float32) * 2 - 1

    def transform(self, results: dict) -> dict:
        
        try:
            import librosa
        except ImportError:
            raise ImportError('Please install librosa first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        if osp.exists(results['audio_path']):
            file_obj = io.BytesIO(self.file_client.get(results['audio_path']))
            y, sr = librosa.load(file_obj, sr=self.sample_rate)
        else:

            pad_func = getattr(self, f'_{self.pad_method}_pad')
            y = pad_func(int(round(10.0 * self.sample_rate)))
            sr = self.sample_rate

        results['length'] = y.shape[0]
        results['sample_rate'] = sr
        results['audios'] = y
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'sample_rate={self.sample_rate}, '
                    f'pad_method={self.pad_method})')
        return repr_str


@TRANSFORMS.register_module()
class LoadAudioFeature(BaseTransform):
    

    def __init__(self, pad_method='zero'):
        if pad_method not in ['zero', 'random']:
            raise NotImplementedError
        self.pad_method = pad_method

    @staticmethod
    def _zero_pad(shape):
        
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape):
        

        return np.random.rand(shape).astype(np.float32)

    def transform(self, results):
        
        if osp.exists(results['audio_path']):
            feature_map = np.load(results['audio_path'])
        else:


            pad_func = getattr(self, f'_{self.pad_method}_pad')
            feature_map = pad_func((640, 80))

        results['length'] = feature_map.shape[0]
        results['audios'] = feature_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pad_method={self.pad_method})')
        return repr_str


@TRANSFORMS.register_module()
class AudioDecode(BaseTransform):
    

    def __init__(self, fixed_length: int = 32000) -> None:
        self.fixed_length = fixed_length

    def transform(self, results: dict) -> dict:
        
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()
        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0])),
                    mode='constant')

            resampled_clips.append(truncated_audio)

        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(fixed_length='{self.fixed_length}')"
        return repr_str


@TRANSFORMS.register_module()
class BuildPseudoClip(BaseTransform):
    

    def __init__(self, clip_len):
        self.clip_len = clip_len

    def transform(self, results):
        

        assert len(results['imgs']) == 1
        im = results['imgs'][0]
        for _ in range(1, self.clip_len):
            results['imgs'].append(np.copy(im))
        results['clip_len'] = self.clip_len
        results['num_clips'] = 1
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@TRANSFORMS.register_module()
class AudioFeatureSelector(BaseTransform):
    

    def __init__(self, fixed_length=128):
        self.fixed_length = fixed_length

    def transform(self, results):
        
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()

        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx, :]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length, :]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0]), (0, 0)),
                    mode='constant')

            resampled_clips.append(truncated_audio)
        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@TRANSFORMS.register_module()
class LoadLocalizationFeature(BaseTransform):
    

    def transform(self, results):
        
        data_path = results['feature_path']
        raw_feature = np.loadtxt(
            data_path, dtype=np.float32, delimiter=',', skiprows=1)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str


@TRANSFORMS.register_module()
class GenerateLocalizationLabels(BaseTransform):
    

    def transform(self, results):
        
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@TRANSFORMS.register_module()
class LoadProposals(BaseTransform):
    

    def __init__(self,
                 top_k,
                 pgm_proposals_dir,
                 pgm_features_dir,
                 proposal_ext='.csv',
                 feature_ext='.npy'):
        self.top_k = top_k
        self.pgm_proposals_dir = pgm_proposals_dir
        self.pgm_features_dir = pgm_features_dir
        valid_proposal_ext = ('.csv', )
        if proposal_ext not in valid_proposal_ext:
            raise NotImplementedError
        self.proposal_ext = proposal_ext
        valid_feature_ext = ('.npy', )
        if feature_ext not in valid_feature_ext:
            raise NotImplementedError
        self.feature_ext = feature_ext

    def transform(self, results):
        
        video_name = results['video_name']
        proposal_path = osp.join(self.pgm_proposals_dir,
                                 video_name + self.proposal_ext)
        if self.proposal_ext == '.csv':
            pgm_proposals = np.loadtxt(
                proposal_path, dtype=np.float32, delimiter=',', skiprows=1)

        pgm_proposals = np.array(pgm_proposals[:self.top_k])
        tmin = pgm_proposals[:, 0]
        tmax = pgm_proposals[:, 1]
        tmin_score = pgm_proposals[:, 2]
        tmax_score = pgm_proposals[:, 3]
        reference_temporal_iou = pgm_proposals[:, 5]

        feature_path = osp.join(self.pgm_features_dir,
                                video_name + self.feature_ext)
        if self.feature_ext == '.npy':
            bsp_feature = np.load(feature_path).astype(np.float32)

        bsp_feature = bsp_feature[:self.top_k, :]
        results['bsp_feature'] = bsp_feature
        results['tmin'] = tmin
        results['tmax'] = tmax
        results['tmin_score'] = tmin_score
        results['tmax_score'] = tmax_score
        results['reference_temporal_iou'] = reference_temporal_iou

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'top_k={self.top_k}, '
                    f'pgm_proposals_dir={self.pgm_proposals_dir}, '
                    f'pgm_features_dir={self.pgm_features_dir}, '
                    f'proposal_ext={self.proposal_ext}, '
                    f'feature_ext={self.feature_ext})')
        return repr_str
