
import random

import mmengine
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.utils import digit_version

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TorchVisionWrapper(BaseTransform):
    

    def __init__(self, op, **kwargs):
        try:
            import torchvision
            import torchvision.transforms as tv_trans
        except ImportError:
            raise RuntimeError('Install torchvision to use TorchvisionTrans')
        if digit_version(torchvision.__version__) < digit_version('0.8.0'):
            raise RuntimeError('The version of torchvision should be at least '
                               '0.8.0')

        trans = getattr(tv_trans, op, None)
        assert trans, f'Transform {op} not in torchvision'
        self.trans = trans(**kwargs)

    def transform(self, results):
        
        assert 'imgs' in results

        imgs = [x.transpose(2, 0, 1) for x in results['imgs']]
        imgs = to_tensor(np.stack(imgs))

        imgs = self.trans(imgs).data.numpy()
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        imgs = imgs.astype(np.uint8)
        imgs = [x.transpose(1, 2, 0) for x in imgs]
        results['imgs'] = imgs
        return results


@TRANSFORMS.register_module()
class PytorchVideoWrapper(BaseTransform):
    

    def __init__(self, op, **kwargs):
        try:
            import pytorchvideo.transforms as ptv_trans
            import torch
        except ImportError:
            raise RuntimeError('Install pytorchvideo to use PytorchVideoTrans')
        if digit_version(torch.__version__) < digit_version('1.8.0'):
            raise RuntimeError(
                'The version of PyTorch should be at least 1.8.0')

        trans = getattr(ptv_trans, op, None)
        assert trans, f'Transform {op} not in pytorchvideo'

        supported_pytorchvideo_trans = ('AugMix', 'RandAugment',
                                        'RandomResizedCrop', 'ShortSideScale',
                                        'RandomShortSideScale')
        assert op in supported_pytorchvideo_trans,\
            f'PytorchVideo Transform {op} is not supported in MMAction2'

        self.trans = trans(**kwargs)
        self.op = op

    def transform(self, results):
        
        assert 'imgs' in results

        assert 'gt_bboxes' not in results,\
            f'PytorchVideo {self.op} doesn\'t support bboxes yet.'
        assert 'proposals' not in results,\
            f'PytorchVideo {self.op} doesn\'t support bboxes yet.'

        if self.op in ('AugMix', 'RandAugment'):

            imgs = [x.transpose(2, 0, 1) for x in results['imgs']]
            imgs = to_tensor(np.stack(imgs))
        else:


            imgs = to_tensor((np.stack(results['imgs']).transpose(3, 0, 1, 2) /
                              255.).astype(np.float32))

        imgs = self.trans(imgs).data.numpy()

        if self.op in ('AugMix', 'RandAugment'):
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)


            imgs = [x.transpose(1, 2, 0) for x in imgs]
        else:

            imgs = imgs * 255
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)


            imgs = [x for x in imgs.transpose(1, 2, 3, 0)]

        results['imgs'] = imgs

        return results


@TRANSFORMS.register_module()
class ImgAug(BaseTransform):
    

    def __init__(self, transforms):
        import imgaug.augmenters as iaa

        if transforms == 'default':
            self.transforms = self.default_transforms()
        elif isinstance(transforms, list):
            assert all(isinstance(trans, dict) for trans in transforms)
            self.transforms = transforms
        elif isinstance(transforms, iaa.Augmenter):
            self.aug = self.transforms = transforms
        else:
            raise ValueError('transforms must be `default` or a list of dicts'
                             ' or iaa.Augmenter object')

        if not isinstance(transforms, iaa.Augmenter):
            self.aug = iaa.Sequential(
                [self.imgaug_builder(t) for t in self.transforms])

    @staticmethod
    def default_transforms():
        

        num_augmenters = 2
        cur_magnitude, max_magnitude = 9, 10
        cur_level = 1.0 * cur_magnitude / max_magnitude

        return [
            dict(
                type='SomeOf',
                n=num_augmenters,
                children=[
                    dict(
                        type='ShearX',
                        shear=17.19 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='ShearY',
                        shear=17.19 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='TranslateX',
                        percent=.2 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='TranslateY',
                        percent=.2 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='Rotate',
                        rotate=30 * cur_level * random.choice([-1, 1])),
                    dict(type='Posterize', nb_bits=max(1, int(4 * cur_level))),
                    dict(type='Solarize', threshold=256 * cur_level),
                    dict(type='EnhanceColor', factor=1.8 * cur_level + .1),
                    dict(type='EnhanceContrast', factor=1.8 * cur_level + .1),
                    dict(
                        type='EnhanceBrightness', factor=1.8 * cur_level + .1),
                    dict(type='EnhanceSharpness', factor=1.8 * cur_level + .1),
                    dict(type='Autocontrast', cutoff=0),
                    dict(type='Equalize'),
                    dict(type='Invert', p=1.),
                    dict(
                        type='Cutout',
                        nb_iterations=1,
                        size=0.2 * cur_level,
                        squared=True)
                ])
        ]

    def imgaug_builder(self, cfg):
        
        import imgaug.augmenters as iaa

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            obj_cls = getattr(iaa, obj_type) if hasattr(iaa, obj_type)\
                else getattr(iaa.pillike, obj_type)
        elif issubclass(obj_type, iaa.Augmenter):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        for aug_list_key in ['children', 'then_list', 'else_list']:
            if aug_list_key in args:
                args[aug_list_key] = [
                    self.imgaug_builder(child) for child in args[aug_list_key]
                ]

        return obj_cls(**args)

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.aug})'
        return repr_str

    def transform(self, results):
        
        assert results['modality'] == 'RGB', 'Imgaug only support RGB images.'
        in_type = results['imgs'][0].dtype

        cur_aug = self.aug.to_deterministic()

        results['imgs'] = [
            cur_aug.augment_image(frame) for frame in results['imgs']
        ]
        img_h, img_w, _ = results['imgs'][0].shape

        out_type = results['imgs'][0].dtype
        assert in_type == out_type,\
            ('Imgaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        if 'gt_bboxes' in results:
            from imgaug.augmentables import bbs
            bbox_list = [
                bbs.BoundingBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                for bbox in results['gt_bboxes']
            ]
            bboxes = bbs.BoundingBoxesOnImage(
                bbox_list, shape=results['img_shape'])
            bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
            results['gt_bboxes'] = [[
                max(bbox.x1, 0),
                max(bbox.y1, 0),
                min(bbox.x2, img_w),
                min(bbox.y2, img_h)
            ] for bbox in bbox_aug.items]
            if 'proposals' in results:
                bbox_list = [
                    bbs.BoundingBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                    for bbox in results['proposals']
                ]
                bboxes = bbs.BoundingBoxesOnImage(
                    bbox_list, shape=results['img_shape'])
                bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
                results['proposals'] = [[
                    max(bbox.x1, 0),
                    max(bbox.y1, 0),
                    min(bbox.x2, img_w),
                    min(bbox.y2, img_h)
                ] for bbox in bbox_aug.items]

        results['img_shape'] = (img_h, img_w)

        return results
