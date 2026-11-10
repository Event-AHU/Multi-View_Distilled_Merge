import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class RawframeDataset(BaseActionDataset):
    

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(img=''),


                 filename_tmpl: str = '{:08}.png',
                 with_offset: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 1,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            video_info = {}
            idx = 0

            frame_dir = line_split[idx]
            if self.data_prefix['img'] is not None:




                frame_dir = osp.join(self.data_prefix['img'], frame_dir)
            video_info['frame_dir'] = frame_dir
            idx += 1
            if self.with_offset:

                video_info['offset'] = int(line_split[idx])
                video_info['total_frames'] = int(line_split[idx + 1])
                idx += 2
            else:

                video_info['total_frames'] = int(line_split[idx])
                idx += 1

            label = [int(x) for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            if self.multi_class:
                assert self.num_classes is not None
                video_info['label'] = label
            else:
                assert len(label) == 1
                video_info['label'] = label[0]
            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        
        data_info = super().get_data_info(idx)
        data_info['filename_tmpl'] = self.filename_tmpl
        return data_info
