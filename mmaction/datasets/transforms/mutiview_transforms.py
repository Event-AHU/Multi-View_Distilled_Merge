import copy as cp
import os
import os.path as osp
from typing import Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from mmaction.registry import TRANSFORMS


def _infer_default_rgb_tmpl(directory_rgb: str) -> str:
    names = [x for x in os.listdir(directory_rgb) if osp.isfile(osp.join(directory_rgb, x))]
    names.sort()
    if not names:
        raise FileNotFoundError(f'No frame files found in {directory_rgb}')
    name = names[0]
    base, ext = osp.splitext(name)
    digits = len(base)
    if digits > 0 and base.isdigit():
        return '{:0' + str(digits) + 'd}' + ext
    raise RuntimeError(f'Cannot infer numeric RGB filename template from {name} in {directory_rgb}')


def _resize_hwc_array(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr

    dtype = arr.dtype
    ten = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()
    ten = F.interpolate(ten, size=(out_h, out_w), mode='bilinear', align_corners=False)
    out = ten.squeeze(0).permute(1, 2, 0).cpu().numpy()

    if np.issubdtype(dtype, np.integer):
        out = np.clip(np.rint(out), 0, np.iinfo(dtype).max).astype(dtype)
    else:
        out = out.astype(dtype, copy=False)
    return out


def _center_crop_hwc_array(arr: np.ndarray, crop_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = int(crop_size[0]), int(crop_size[1])

    h, w = arr.shape[:2]
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    bottom = min(top + crop_h, h)
    right = min(left + crop_w, w)
    return arr[top:bottom, left:right, ...]


def _flip_hwc_array(arr: np.ndarray, direction: str) -> np.ndarray:
    if direction == 'horizontal':
        return np.ascontiguousarray(arr[:, ::-1, ...])
    if direction == 'vertical':
        return np.ascontiguousarray(arr[::-1, :, ...])
    if direction == 'diagonal':
        return np.ascontiguousarray(arr[::-1, ::-1, ...])
    raise ValueError(f'Unsupported flip direction: {direction}')


@TRANSFORMS.register_module()
class MutiviewKeepOnlyRGBFromConcat(BaseTransform):
    

    def transform(self, results: dict) -> dict:
        imgs = results.get('imgs', None)
        imgs_event = results.get('imgs_event', None)
        if imgs is None or imgs_event is None:
            return results

        event_len = len(imgs_event)
        if event_len == 0:
            return results
        if len(imgs) < event_len:
            raise RuntimeError(f'imgs length {len(imgs)} < imgs_event length {event_len}')

        results['imgs'] = [cp.deepcopy(x) for x in imgs[:event_len]]
        return results


@TRANSFORMS.register_module()
class MutiviewUseStreamFramesAsImgsEvent(BaseTransform):
    

    def __init__(self, pop_src: bool = False):
        self.pop_src = pop_src

    def transform(self, results: dict) -> dict:
        stream_frames = results.get('stream_frame_all', None)
        if stream_frames is None:
            raise KeyError('`stream_frame_all` not found in results.')
        results['imgs_event'] = stream_frames
        if self.pop_src:
            results.pop('stream_frame_all', None)
        return results


@TRANSFORMS.register_module()
class MutiviewResizeImgsEventToMatchImgs(BaseTransform):
    

    def transform(self, results: dict) -> dict:
        imgs = results.get('imgs', None)
        imgs_event = results.get('imgs_event', None)
        if imgs is None or imgs_event is None or len(imgs) == 0 or len(imgs_event) == 0:
            return results

        target_h, target_w = imgs[0].shape[:2]
        results['imgs_event'] = [
            _resize_hwc_array(np.asarray(img), target_h, target_w)
            for img in results['imgs_event']
        ]
        return results


@TRANSFORMS.register_module()
class MutiviewApplyCropToImgsEvent(BaseTransform):
    

    def transform(self, results: dict) -> dict:
        if 'imgs_event' not in results or 'crop_bbox' not in results:
            return results

        left, top, right, bottom = [int(x) for x in results['crop_bbox']]
        results['imgs_event'] = [
            np.asarray(img)[top:bottom, left:right, ...] for img in results['imgs_event']
        ]
        return results


@TRANSFORMS.register_module()
class MutiviewCenterCropImgsEvent(BaseTransform):
    

    def __init__(self, crop_size: Union[int, Tuple[int, int]]):
        self.crop_size = crop_size

    def transform(self, results: dict) -> dict:
        if 'imgs_event' not in results:
            return results
        results['imgs_event'] = [
            _center_crop_hwc_array(np.asarray(img), self.crop_size)
            for img in results['imgs_event']
        ]
        return results


@TRANSFORMS.register_module()
class MutiviewFlipImgsEvent(BaseTransform):
    

    def transform(self, results: dict) -> dict:
        if 'imgs_event' not in results:
            return results
        if not results.get('flip', False):
            return results
        direction = results.get('flip_direction', 'horizontal')
        results['imgs_event'] = [
            _flip_hwc_array(np.asarray(img), direction) for img in results['imgs_event']
        ]
        return results


@TRANSFORMS.register_module()
class MutiviewLoadRGBWithVoxel(BaseTransform):
    

    def __init__(self,
                 rgb_root: str,
                 voxel_root: str,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 rgb_filename_tmpl: Optional[str] = '{:04d}.png',
                 rgb_index_offset: int = 0,
                 voxel_feat_dim: int = 16,
                 voxel_accumulate: str = 'max'):
        self.rgb_root = rgb_root
        self.voxel_root = voxel_root
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.file_client_args = dict(file_client_args or dict(io_backend=io_backend))
        self.rgb_filename_tmpl = rgb_filename_tmpl
        self.rgb_index_offset = int(rgb_index_offset)
        self.voxel_feat_dim = int(voxel_feat_dim)
        self.voxel_accumulate = voxel_accumulate
        self.file_client = None

    def _get_file_client(self):
        if getattr(self, 'file_client', None) is None:
            fc_args = dict(self.file_client_args) if self.file_client_args is not None else {}

            backend = fc_args.pop('backend', None)
            if backend is None:
                backend = fc_args.pop('io_backend', 'disk')

            self.file_client = FileClient(backend=backend, **fc_args)
        return self.file_client

    def _load_rgb_frames(self, directory_rgb: str, frame_inds_0: np.ndarray, offset: int):
        file_client = self._get_file_client()
        mmcv.use_backend(self.decoding_backend)

        tmpl = self.rgb_filename_tmpl
        if tmpl is None:
            tmpl = _infer_default_rgb_tmpl(directory_rgb)

        imgs = []
        cache = {}
        for i, idx0 in enumerate(frame_inds_0):
            if idx0 in cache:
                imgs.append(cp.deepcopy(imgs[cache[idx0]]))
                continue
            cache[idx0] = i

            frame_id = int(idx0 + offset + self.rgb_index_offset)
            filepath = osp.join(directory_rgb, tmpl.format(frame_id))
            img_bytes = file_client.get(filepath)
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            imgs.append(cur_frame)
        return imgs

    def _load_voxel_data(self, voxel_path: str):
        return torch.load(voxel_path, map_location='cpu')

    def _voxel_data_to_dense_volume(self, voxel_data) -> np.ndarray:
        x = getattr(voxel_data, 'x', None)
        pos = getattr(voxel_data, 'pos', None)
        if x is None or pos is None:
            if isinstance(voxel_data, dict):
                x = voxel_data.get('x', None)
                pos = voxel_data.get('pos', None)
        if x is None or pos is None:
            raise KeyError('Voxel data must contain `x` and `pos`.')

        x = torch.as_tensor(x).float().cpu()
        pos = torch.as_tensor(pos).cpu()
        if pos.ndim != 2 or pos.size(1) < 3:
            raise RuntimeError(f'Expect voxel_data.pos as [N,3], got shape={tuple(pos.shape)}')
        if x.ndim != 2:
            raise RuntimeError(f'Expect voxel_data.x as [N,C], got shape={tuple(x.shape)}')


        if pos.dtype.is_floating_point:
            pos = pos.round().long()
        else:
            pos = pos.long()


        t_idx = pos[:, 0]
        x_idx = pos[:, 1]
        y_idx = pos[:, 2]


        t_idx = t_idx - int(t_idx.min().item())
        x_idx = x_idx - int(x_idx.min().item())
        y_idx = y_idx - int(y_idx.min().item())

        feat = x
        if feat.size(1) < self.voxel_feat_dim:
            pad = torch.zeros(feat.size(0), self.voxel_feat_dim - feat.size(1), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=1)
        elif feat.size(1) > self.voxel_feat_dim:
            feat = feat[:, :self.voxel_feat_dim]

        t_bins = int(t_idx.max().item()) + 1
        w_bins = int(x_idx.max().item()) + 1
        h_bins = int(y_idx.max().item()) + 1
        volume = np.zeros((t_bins, h_bins, w_bins, self.voxel_feat_dim), dtype=np.float32)

        t_np = t_idx.numpy()
        x_np = x_idx.numpy()
        y_np = y_idx.numpy()
        feat_np = feat.numpy().astype(np.float32, copy=False)

        if self.voxel_accumulate == 'sum':
            for i in range(feat_np.shape[0]):
                volume[t_np[i], y_np[i], x_np[i], :] += feat_np[i]
        else:
            for i in range(feat_np.shape[0]):
                volume[t_np[i], y_np[i], x_np[i], :] = np.maximum(
                    volume[t_np[i], y_np[i], x_np[i], :], feat_np[i])
        return volume

    @staticmethod
    def _select_time_indices(total_t: int, target_len: int, rgb_frame_inds_0: np.ndarray, total_rgb: int) -> np.ndarray:
        if total_t <= 0:
            raise RuntimeError(f'Invalid voxel time bins: {total_t}')
        if target_len <= 0:
            raise RuntimeError(f'Invalid target_len: {target_len}')

        if total_rgb <= 1:
            if total_t == 1:
                return np.zeros((target_len,), dtype=np.int64)
            return np.linspace(0, total_t - 1, num=target_len).round().astype(np.int64)

        mapped = np.round(rgb_frame_inds_0.astype(np.float32) / float(total_rgb - 1) * float(total_t - 1))
        mapped = np.clip(mapped.astype(np.int64), 0, total_t - 1)
        return mapped

    def transform(self, results: dict) -> dict:
        directory_tsc = results['frame_dir']
        vid1, vid2 = directory_tsc.split('/')[-2], directory_tsc.split('/')[-1]
        directory_rgb = osp.join(self.rgb_root, vid1, vid2)
        voxel_path = osp.join(self.voxel_root, vid1, f'{vid2}.pt')

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
        imgs = self._load_rgb_frames(directory_rgb, frame_inds_0, offset)

        rgb_names = [x for x in os.listdir(directory_rgb) if osp.isfile(osp.join(directory_rgb, x))]
        rgb_names.sort()
        total_rgb = len(rgb_names)
        if total_rgb == 0:
            raise FileNotFoundError(f'No RGB frames in {directory_rgb}')

        voxel_data = self._load_voxel_data(voxel_path)
        volume = self._voxel_data_to_dense_volume(voxel_data)
        voxel_inds = self._select_time_indices(
            total_t=volume.shape[0],
            target_len=len(frame_inds_0),
            rgb_frame_inds_0=frame_inds_0,
            total_rgb=total_rgb)
        imgs_event = [np.ascontiguousarray(volume[int(ti)]) for ti in voxel_inds]

        results['imgs'] = imgs
        results['imgs_event'] = imgs_event
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['voxel_event_shape'] = volume.shape
        return results
