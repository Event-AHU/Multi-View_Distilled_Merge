
import torch
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer
import pdb

@MODELS.register_module()
class Recognizer3D(BaseRecognizer):
    

    def extract_feat(self,
                     inputs: Tensor,
                     imgs_event: Tensor,
                     stage: str = 'neck',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> tuple:
        


        loss_predict_kwargs = dict()

        num_segs = inputs.shape[1]







        inputs = inputs.view((-1, ) + inputs.shape[2:])

        imgs_event = imgs_event.view((-1, ) + imgs_event.shape[2:])


        if test_mode:

            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get('fcn_test', False)

            if self.test_cfg is not None and self.test_cfg.get('max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0]
                assert num_segs == total_views, ('max_testing_views is only compatible with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    batch_imgs_event = imgs_event[view_ptr:view_ptr + max_testing_views]

                    imgs_list=[batch_imgs, batch_imgs_event]

                    feat = self.backbone(imgs_list)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views


                def recursively_cat(feats):
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x = recursively_cat(feats)
                else:
                    x = torch.cat(feats)
            else:


                imgs_list=[inputs, imgs_event]

                x = self.backbone(imgs_list)
                if self.with_neck:
                    x, _ = self.neck(x)

            return x, loss_predict_kwargs
        else:



            imgs_list=[inputs, imgs_event]

            x = self.backbone(imgs_list)
            if stage == 'backbone':
                return x, loss_predict_kwargs

            loss_aux = dict()
            if self.with_neck:
                x, loss_aux = self.neck(x, data_samples=data_samples)


            loss_predict_kwargs['loss_aux'] = loss_aux
            if stage == 'neck':
                return x, loss_predict_kwargs


            if self.with_cls_head and stage == 'head':
                x = self.cls_head(x, **loss_predict_kwargs)
                return x, loss_predict_kwargs
