
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.fileio import list_from_file
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmaction.registry import INFERENCERS
from mmaction.structures import ActionDataSample
from mmaction.utils import ConfigType

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='action-recognition')
@INFERENCERS.register_module()
class ActionRecogInferencer(BaseInferencer):
    

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'vid_out_dir', 'draw_pred', 'fps',
        'out_type', 'target_resolution'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample'
    }

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 label_file: Optional[str] = None,
                 input_format: str = 'video',
                 pack_cfg: dict = {},
                 scope: Optional[str] = 'mmaction') -> None:


        self.num_visualized_vids = 0
        self.input_format = input_format
        self.pack_cfg = pack_cfg.copy()
        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        if label_file is not None:
            self.visualizer.dataset_meta = dict(
                classes=list_from_file(label_file))

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 vid_out_dir: str = '',
                 out_type: str = 'video',
                 print_result: bool = False,
                 pred_out_file: str = '',
                 target_resolution: Optional[Tuple[int]] = None,
                 **kwargs) -> dict:
        
        return super().__call__(
            inputs,
            return_datasamples,
            batch_size,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            vid_out_dir=vid_out_dir,
            print_result=print_result,
            pred_out_file=pred_out_file,
            out_type=out_type,
            target_resolution=target_resolution,
            **kwargs)

    def _inputs_to_list(self, inputs: InputsType) -> list:
        
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        
        test_pipeline = cfg.test_dataloader.dataset.pipeline

        if self.input_format == 'array':
            for i in range(len(test_pipeline)):
                if 'Decode' in test_pipeline[i]['type']:
                    test_pipeline[i] = dict(type='ArrayDecode')
            test_pipeline = [
                x for x in test_pipeline if 'Init' not in x['type']
            ]
        elif self.input_format == 'video':
            if 'Init' not in test_pipeline[0]['type']:
                test_pipeline = [dict(type='DecordInit')] + test_pipeline
            else:
                test_pipeline[0] = dict(type='DecordInit')
            for i in range(len(test_pipeline)):
                if 'Decode' in test_pipeline[i]['type']:
                    test_pipeline[i] = dict(type='DecordDecode')
        elif self.input_format == 'rawframes':
            if 'Init' in test_pipeline[0]['type']:
                test_pipeline = test_pipeline[1:]
            for i in range(len(test_pipeline)):
                if 'Decode' in test_pipeline[i]['type']:
                    test_pipeline[i] = dict(type='RawFrameDecode')


        for i in range(len(test_pipeline)):
            if test_pipeline[i]['type'] in ['ThreeCrop', 'TenCrop']:
                test_pipeline[i]['type'] = 'CenterCrop'

        if cfg.model.type == 'Recognizer3D':
            for i in range(len(test_pipeline)):
                if test_pipeline[i]['type'] == 'SampleFrames':
                    test_pipeline[i]['num_clips'] = 1

        test_pipeline.insert(
            0,
            dict(
                type='InferencerPackInput',
                input_format=self.input_format,
                **self.pack_cfg))

        return Compose(test_pipeline)

    def visualize(
        self,
        inputs: InputsType,
        preds: PredType,
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = 0,
        draw_pred: bool = True,
        fps: int = 30,
        out_type: str = 'video',
        target_resolution: Optional[Tuple[int]] = None,
        vid_out_dir: str = '',
    ) -> Union[List[np.ndarray], None]:
        
        if self.visualizer is None or (not show and vid_out_dir == ''
                                       and not return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                frames = single_input
                video_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                frames = single_input.copy()
                video_num = str(self.num_visualized_vids).zfill(8)
                video_name = f'{video_num}.mp4'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_path = osp.join(vid_out_dir, video_name) if vid_out_dir != ''\
                else None

            visualization = self.visualizer.add_datasample(
                video_name,
                frames,
                pred,
                show_frames=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                fps=fps,
                out_type=out_type,
                out_path=out_path,
                target_resolution=target_resolution,
            )
            results.append(visualization)
            self.num_visualized_vids += 1

        return results

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        pred_out_file: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        
        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred)
                results.append(result)

        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict

    def pred2dict(self, data_sample: ActionDataSample) -> Dict:
        
        result = {}
        result['pred_labels'] = data_sample.pred_labels.item.tolist()
        result['pred_scores'] = data_sample.pred_scores.item.tolist()
        return result
