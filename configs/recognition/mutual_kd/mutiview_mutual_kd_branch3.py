import os

_base_ = ['../../_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmaction.datasets.transforms.formatting_mutual_kd',
        'mmaction.datasets.transforms.loading_sst_stream',
        'mmaction.datasets.transforms.mutiview_transforms',
        'mmaction.models.backbones.mutiview_tscformer',
        'mmaction.models.recognizers.mutiview_recognizer2d',
        'mmaction.models.recognizers.mutiview_logits_mutual_distiller',
    ],
    allow_failed_imports=False)

dataset_type = 'RawframeDataset'
tsc_data_root = os.getenv('RGB_EVENT_ROOT', '')
sst_rgb_root = os.getenv('RGB_FRAME_ROOT', '')
sst_stream_root = os.getenv('STREAM_ROOT', '')
voxel_root = os.getenv('VOXEL_ROOT', '')
ann_file_train = os.getenv('TRAIN_ANN_FILE', '')
ann_file_val = os.getenv('VAL_ANN_FILE', '')
ann_file_test = os.getenv('TEST_ANN_FILE', '')

file_client_args = dict()

train_pipeline = [

    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),


    dict(type='RawFrameDecode'),
    dict(type='MutiviewKeepOnlyRGBFromConcat'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='MultiScaleCrop',
         input_size=224,
         scales=(1, 0.875, 0.75, 0.66),
         random_crop=False,
         max_wh_scale_gap=1,
         num_fixed_crops=13),
    dict(type='MutiviewApplyCropToImgsEvent'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='MutiviewFlipImgsEvent'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='SaveToKey', src='imgs', dst='mv_frame_rgb', pop_src=True),
    dict(type='MoveKey', src='imgs_event', dst='mv_frame_event', pop_src=True),


    dict(
        type='RawFrameDecodeSSTWithStream',
        sst_rgb_root=sst_rgb_root,
        stream_root=sst_stream_root,
        io_backend='disk',
        decoding_backend='cv2',
        align_stream_with_rgb=True,
        file_client_args=file_client_args,
        sst_filename_tmpl='{:04d}.png',
        sst_index_offset=0),
    dict(type='MutiviewUseStreamFramesAsImgsEvent', pop_src=True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='MultiScaleCrop',
         input_size=224,
         scales=(1, 0.875, 0.75, 0.66),
         random_crop=False,
         max_wh_scale_gap=1,
         num_fixed_crops=13),
    dict(type='MutiviewApplyCropToImgsEvent'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='MutiviewFlipImgsEvent'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='SaveToKey', src='imgs', dst='mv_stream_rgb', pop_src=True),
    dict(type='MoveKey', src='imgs_event', dst='mv_stream_event', pop_src=True),


    dict(
        type='MutiviewLoadRGBWithVoxel',
        rgb_root=sst_rgb_root,
        voxel_root=voxel_root,
        io_backend='disk',
        decoding_backend='cv2',
        file_client_args=file_client_args,
        rgb_filename_tmpl='{:04d}.png',
        rgb_index_offset=0,
        voxel_feat_dim=16,
        voxel_accumulate='max'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='MultiScaleCrop',
         input_size=224,
         scales=(1, 0.875, 0.75, 0.66),
         random_crop=False,
         max_wh_scale_gap=1,
         num_fixed_crops=13),
    dict(type='MutiviewApplyCropToImgsEvent'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='MutiviewFlipImgsEvent'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='SaveToKey', src='imgs', dst='mv_voxel_rgb', pop_src=True),
    dict(type='MoveKey', src='imgs_event', dst='mv_voxel_event', pop_src=True),

    dict(type='PackMutualKDInputs', collect_keys=(
        'mv_frame_rgb', 'mv_frame_event',
        'mv_stream_rgb', 'mv_stream_event',
        'mv_voxel_rgb', 'mv_voxel_event')),
]

val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),

    dict(type='RawFrameDecode'),
    dict(type='MutiviewKeepOnlyRGBFromConcat'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='MutiviewCenterCropImgsEvent', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='SaveToKey', src='imgs', dst='mv_frame_rgb', pop_src=True),
    dict(type='MoveKey', src='imgs_event', dst='mv_frame_event', pop_src=True),


    dict(
        type='RawFrameDecodeSSTWithStream',
        sst_rgb_root=sst_rgb_root,
        stream_root=sst_stream_root,
        io_backend='disk',
        decoding_backend='cv2',
        align_stream_with_rgb=True,
        file_client_args=file_client_args,
        sst_filename_tmpl='{:04d}.png',
        sst_index_offset=0),
    dict(type='MutiviewUseStreamFramesAsImgsEvent', pop_src=True),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='MutiviewCenterCropImgsEvent', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='SaveToKey', src='imgs', dst='mv_stream_rgb', pop_src=True),
    dict(type='MoveKey', src='imgs_event', dst='mv_stream_event', pop_src=True),


    dict(
        type='MutiviewLoadRGBWithVoxel',
        rgb_root=sst_rgb_root,
        voxel_root=voxel_root,
        io_backend='disk',
        decoding_backend='cv2',
        file_client_args=file_client_args,
        rgb_filename_tmpl='{:04d}.png',
        rgb_index_offset=0,
        voxel_feat_dim=16,
        voxel_accumulate='max'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MutiviewResizeImgsEventToMatchImgs'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='MutiviewCenterCropImgsEvent', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='SaveToKey', src='imgs', dst='mv_voxel_rgb', pop_src=True),
    dict(type='MoveKey', src='imgs_event', dst='mv_voxel_event', pop_src=True),

    dict(type='PackMutualKDInputs', collect_keys=(
        'mv_frame_rgb', 'mv_frame_event',
        'mv_stream_rgb', 'mv_stream_event',
        'mv_voxel_rgb', 'mv_voxel_event')),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=10,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=tsc_data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=tsc_data_root),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=tsc_data_root),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

common_head = dict(
    type='TSMHead',
    num_classes=114,
    in_channels=2240,
    num_segments=16,
    spatial_type='avg',
    consensus=dict(type='AvgConsensus', dim=1),
    dropout_ratio=0.5,
    init_std=0.001,
    is_shift=True,
    average_clips='prob',
)

mv_frame_branch = dict(
    type='MutiviewRecognizer2D',
    backbone=dict(
        type='MutiviewTSCFormer',
        pretrained='torchvision://resnet50',
        is_shift=True,
        num_segments=8,
        depth=50,
        norm_eval=False,
        shift_div=8,
        event_in_channels=3,
        event_adapter_out_channels=64,
        init_event_adapter_from_resnet=True,
    ),
    cls_head=common_head,
    train_cfg=None,
    test_cfg=dict(average_clips='prob')
)

mv_stream_branch = dict(
    type='MutiviewRecognizer2D',
    backbone=dict(
        type='MutiviewTSCFormer',
        pretrained='torchvision://resnet50',
        is_shift=True,
        num_segments=8,
        depth=50,
        norm_eval=False,
        shift_div=8,
        event_in_channels=12,
        event_adapter_out_channels=64,
        init_event_adapter_from_resnet=True,
    ),
    cls_head=common_head,
    train_cfg=None,
    test_cfg=dict(average_clips='prob')
)

mv_voxel_branch = dict(
    type='MutiviewRecognizer2D',
    backbone=dict(
        type='MutiviewTSCFormer',
        pretrained='torchvision://resnet50',
        is_shift=True,
        num_segments=8,
        depth=50,
        norm_eval=False,
        shift_div=8,
        event_in_channels=16,
        event_adapter_out_channels=64,
        init_event_adapter_from_resnet=True,
    ),
    cls_head=common_head,
    train_cfg=None,
    test_cfg=dict(average_clips='prob')
)

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    preprocessors=dict(
        mv_frame_rgb=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCHW'),
        mv_frame_event=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCHW'),
        mv_stream_rgb=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCHW'),
        mv_stream_event=dict(
            type='ActionDataPreprocessor',
            mean=None,
            std=None,
            format_shape='NCHW'),
        mv_voxel_rgb=dict(
            type='ActionDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape='NCHW'),
        mv_voxel_event=dict(
            type='ActionDataPreprocessor',
            mean=None,
            std=None,
            format_shape='NCHW'),
    )
)

model = dict(
    type='MutiviewLogitsMutualDistiller',

    branch_devices=dict(zip(
        ('mv_frame', 'mv_stream', 'mv_voxel'),
        os.getenv('MV_BRANCH_DEVICES', 'cuda:0,cuda:1,cuda:2').split(','))),
    data_preprocessor=data_preprocessor,
    kd_temperature=4.0,
    kd_weight=0.1,
    ce_weight=0.6,
    kd_start_epoch=6,
    pred_branch='mv_frame',

    feature_kd_weight=0.15,
    feature_kd_start_epoch=8,
    feature_loss_type='cosine',
    feature_stage='backbone',
    feature_projector_hidden=1024,
    feature_projector_out=512,
    feature_projector_dropout=0.1,
    peer_weight_hidden=256,
    feature_relation_weight=0.15,

    class_prior=None,
    class_prior_weight=0.0,
    branches=dict(
        mv_frame=mv_frame_branch,
        mv_stream=mv_stream_branch,
        mv_voxel=mv_voxel_branch,
    ),
    io_map=dict(
        mv_frame=dict(input='mv_frame_rgb', event='mv_frame_event'),
        mv_stream=dict(input='mv_stream_rgb', event='mv_stream_event'),
        mv_voxel=dict(input='mv_voxel_rgb', event='mv_voxel_event'),
    )
)

default_hooks = dict(
    checkpoint=dict(
        interval=3,
        max_keep_ckpts=10,
        save_best=['acc/mv_frame_top1', 'acc/mv_stream_top1', 'acc/mv_voxel_top1'],
        rule='greater',
    ),
    logger=dict(type='LoggerHook', interval=200),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_begin=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[50, 80], gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0008, momentum=0.9, weight_decay=0.00005),
    clip_grad=dict(max_norm=20, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=128)
