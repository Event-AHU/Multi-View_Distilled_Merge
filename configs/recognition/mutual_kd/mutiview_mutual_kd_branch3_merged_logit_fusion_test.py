_base_ = ['./mutiview_mutual_kd_branch3.py']

model = dict(

    branch_devices=dict(mv_frame='cuda:0', mv_stream='cuda:0', mv_voxel='cuda:0'),


    enable_logit_fusion=True,
    logit_fusion_name='fused',


    logit_fusion_weights=dict(
        mv_frame=0.15,
        mv_stream=0.05,
        mv_voxel=0.80,
    ),
    logit_fusion_temperatures=dict(
        mv_frame=1.0,
        mv_stream=1.0,
        mv_voxel=1.0,
    ),
)