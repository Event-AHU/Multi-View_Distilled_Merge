from typing import Union, Tuple

from mmengine.model import BaseDataPreprocessor

from mmaction.models.data_preprocessors.data_preprocessor import ActionDataPreprocessor
from mmaction.registry import MODELS


@MODELS.register_module()
class TwoStreamActionDataPreprocessor(BaseDataPreprocessor):
    

    def __init__(self,
                 rgb_mean=None,
                 rgb_std=None,
                 event_mean=None,
                 event_std=None,
                 to_rgb=False,
                 format_shape='NCHW'):
        super().__init__()
        self.rgb_preprocessor = ActionDataPreprocessor(
            mean=rgb_mean,
            std=rgb_std,
            to_rgb=to_rgb,
            format_shape=format_shape)

        self.event_preprocessor = ActionDataPreprocessor(
            mean=event_mean,
            std=event_std,
            to_rgb=False,
            format_shape=format_shape)

    def forward(self,
                data: Union[dict, Tuple[dict]],
                training: bool = False):
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        elif isinstance(data, tuple):
            outputs = []
            for data_sample in data:
                outputs.append(self.forward_onesample(data_sample, training=training))
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, training: bool = False):
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs, data_samples = self.rgb_preprocessor.preprocess(
            inputs, data_samples, training)
        data['inputs'] = inputs

        if 'imgs_event' in data and data['imgs_event'] is not None:
            imgs_event = data['imgs_event']
            imgs_event, data_samples = self.event_preprocessor.preprocess(
                imgs_event, data_samples, training)
            data['imgs_event'] = imgs_event

        data['data_samples'] = data_samples
        return data
