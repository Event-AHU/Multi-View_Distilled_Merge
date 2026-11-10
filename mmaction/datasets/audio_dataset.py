
import os.path as osp
from typing import Callable, List, Optional, Union

import torch
from mmengine.utils import check_file_exist

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class AudioDataset(BaseActionDataset):
    

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(audio=''),
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 suffix: str = '.wav',
                 **kwargs) -> None:
        self.suffix = suffix
        super().__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            modality='Audio',
            **kwargs)

    def load_data_list(self) -> List[dict]:
        
        check_file_exist(self.ann_file)
        data_list = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                filename = line_split[idx]
                if self.data_prefix['audio'] is not None:
                    if not filename.endswith(self.suffix):
                        filename = osp.join(self.data_prefix['audio'],
                                            filename + self.suffix)
                    else:
                        filename = osp.join(self.data_prefix['audio'],
                                            filename)
                video_info['audio_path'] = filename
                idx += 1

                video_info['total_frames'] = int(line_split[idx])
                idx += 1

                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                    video_info['label'] = onehot
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                data_list.append(video_info)

        return data_list
