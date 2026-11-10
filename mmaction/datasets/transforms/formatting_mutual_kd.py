
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample
from mmengine.structures import LabelData

@TRANSFORMS.register_module()
class EmptyImgsEvent(BaseTransform):
    
    def transform(self, results: dict) -> dict:

        results['imgs_event'] = []
        return results

@TRANSFORMS.register_module()
class SetKey(BaseTransform):
    
    def __init__(self, key: str, value: str = 'dummy_event_6d'):
        self.key = key
        self.value = value

    def transform(self, results: dict) -> dict:
        if self.value == 'dummy_event_6d':

            results[self.key] = np.zeros((1, 1, 1, 1, 1, 1), dtype=np.uint8)
        elif self.value == 'empty_ndarray':
            results[self.key] = np.empty((0,), dtype=np.float32)
        else:
            results[self.key] = self.value
        return results

@TRANSFORMS.register_module()
class MoveKey(BaseTransform):
    
    def __init__(self, src: str, dst: str, pop_src: bool = True):
        self.src = src
        self.dst = dst
        self.pop_src = pop_src

    def transform(self, results):
        if self.src in results:
            results[self.dst] = results[self.src]
            if self.pop_src:
                results.pop(self.src)
        return results

@TRANSFORMS.register_module()
class ClearKeys(BaseTransform):
    
    def __init__(self, keys):
        self.keys = list(keys)

    def transform(self, results):
        for k in self.keys:
            if k in results:
                results.pop(k)
        return results

@TRANSFORMS.register_module()
class SaveToKey(BaseTransform):
    
    def __init__(self, src: str = 'imgs', dst: str = 'tsc', pop_src: bool = False):
        self.src = src
        self.dst = dst
        self.pop_src = pop_src

    def transform(self, results: dict) -> dict:
        results[self.dst] = results[self.src]
        if self.pop_src:
            results.pop(self.src, None)
        return results

@TRANSFORMS.register_module()
class StackStreamFrames(BaseTransform):
    

    def __init__(self, src='stream_frame_all', dst='sst_stream', dtype='float32'):
        self.src = src
        self.dst = dst
        self.dtype = dtype

    def transform(self, results: dict) -> dict:
        stream_frames = results.get(self.src, None)
        if stream_frames is None:
            raise KeyError(f'`{self.src}` not found in results.')





        if isinstance(stream_frames, list):
            if len(stream_frames) == 1 and isinstance(stream_frames[0], np.ndarray) and stream_frames[0].ndim == 4:
                arr = stream_frames[0]
            else:
                arr = np.stack(stream_frames, axis=0)
        else:
            arr = stream_frames if isinstance(stream_frames, np.ndarray) else np.asarray(stream_frames)


        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim != 4:
            raise ValueError(f'Expect stream frames as (T,H,W,C), but got shape={arr.shape}')

        if self.dtype is not None:
            arr = arr.astype(self.dtype, copy=False)

        results[self.dst] = arr
        return results


@TRANSFORMS.register_module()
class PackMutualKDInputs(BaseTransform):
    
    def __init__(self,
                 collect_keys,
                 meta_keys=('frame_dir', 'total_frames', 'filename_tmpl', 'modality',
                            'start_index', 'offset', 'img_shape', 'original_shape')):
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed = {}


        inputs = {}
        if isinstance(self.collect_keys, (list, tuple)):
            for k in self.collect_keys:
                if k not in results:
                    raise KeyError(f'Key `{k}` not found in results for PackMutualKDInputs.')
                inputs[k] = to_tensor(results[k])
        elif isinstance(self.collect_keys, dict):
            for out_name, res_key in self.collect_keys.items():
                if res_key not in results:
                    raise KeyError(f'Key `{res_key}` not found in results for PackMutualKDInputs.')
                inputs[out_name] = to_tensor(results[res_key])
        else:
            raise TypeError('collect_keys must be a list/tuple or a dict mapping.')

        packed['inputs'] = inputs


        data_sample = ActionDataSample()

        if 'label' in results:

            label = results['label']
            data_sample.gt_labels = LabelData(item=to_tensor(label))

        metainfo = {}
        for k in self.meta_keys:
            if k in results:
                metainfo[k] = results[k]
        if metainfo:
            data_sample.set_metainfo(metainfo)

        packed['data_samples'] = data_sample
        return packed
