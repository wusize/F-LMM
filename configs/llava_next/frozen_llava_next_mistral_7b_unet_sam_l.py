# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from torch.nn import GroupNorm, ReLU
from transformers import AutoTokenizer
from xtuner.engine.runner import TrainLoop

from mmengine.dataset import DefaultSampler
from src.datasets.gcg import (GCGDataset, FlickrForGCGDataset, RefCOCOGForGCGDataset,
                              concat_datasets, gcg_collate_fn)
from src.datasets.png import PNGDataset
from src.models.llava_next.modeling_llava_next import CustomLlavaNextForConditionalGeneration
from src.datasets.llava_next_processors import CustomLlavaNextImageProcessor
# from src.models.meta_arch import FrozenLlava
from src.models.frozen_llava_next import FrozenLlavaNextSAM
from src.models.mask_heads import UNetHead
from xtuner.utils.templates import PROMPT_TEMPLATE
from src.models.segment_modules.sam_wrapper import SAMWrapper
from mmdet.models import DiceLoss, CrossEntropyLoss
from mmseg.models.backbones.unet import InterpConv
from src.models.key_phrase_heads import KeyPhraseHead

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 1
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)



#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# Model
prompt_template = PROMPT_TEMPLATE.mistral
llava_name = 'llava-hf/llava-v1.6-mistral-7b-hf'
unet = dict(type=UNetHead,
            upsample_input=64,
            in_channels=2048,
            base_channels=64,
            num_stages=4,
            strides=(1, 1, 1, 1),
            enc_num_convs=(2, 2, 2, 2),   # the first enc is for projection
            dec_num_convs=(2, 2, 2),
            downsamples=(True, True, True),
            enc_dilations=(1, 1, 1, 1),
            dec_dilations=(1, 1, 1),
            norm_cfg=dict(type=GroupNorm, num_groups=1),
            upsample_cfg=dict(type=InterpConv)
            )
loss_mask = dict(
    type=CrossEntropyLoss,
    use_sigmoid=True,
    reduction='mean',
    loss_weight=1.0)
loss_dice = dict(
    type=DiceLoss,
    use_sigmoid=True,
    activate=True,
    reduction='mean',
    naive_dice=True,
    eps=1.0,
    loss_weight=1.0)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llava_name)
image_processor = dict(
    type=CustomLlavaNextImageProcessor.from_pretrained,
    pretrained_model_name_or_path=llava_name)

model = dict(
    type=FrozenLlavaNextSAM,
    sam=dict(type=SAMWrapper,
             use_text=True, use_mask=True, multimask_output=False,
             model_name='vit_l', checkpoint='checkpoints/sam_vit_l_0b3195.pth',),
    model=dict(type=CustomLlavaNextForConditionalGeneration.from_pretrained,
               pretrained_model_name_or_path=llava_name,
               torch_dtype=torch.float16, low_cpu_mem_usage=True),
    mask_head=unet,
    loss_mask=loss_mask,
    loss_dice=loss_dice,
    key_phrase_head=dict(type=KeyPhraseHead,
                         encoder=dict(  # DetrTransformerEncoder
                             num_layers=3,
                             layer_cfg=dict(  # DetrTransformerEncoderLayer
                                 self_attn_cfg=dict(  # MultiheadAttention
                                     embed_dims=256,
                                     num_heads=8,
                                     dropout=0.1,
                                     batch_first=True),
                                 ffn_cfg=dict(
                                     embed_dims=256,
                                     feedforward_channels=2048,
                                     num_fcs=2,
                                     ffn_drop=0.1,
                                     act_cfg=dict(type=ReLU, inplace=True)))),
                         loss_mask=loss_mask,
                         loss_dice=loss_dice)
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

datasets_list = [
    dict(type=GCGDataset,
         ceph_path='BJ17:S3://wusize/GranDf_HA_images/train',
         json_file='data/GranDf_HA_GCG_train.json',
         local_path='data/GranDf_HA_images/train',
         prompt_template=prompt_template,
         tokenizer=tokenizer,
         image_processor=image_processor),
    dict(type=GCGDataset,
         ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco',
         json_file='data/OpenPsgGCG_train.json',
         local_path='data/coco',
         prompt_template=prompt_template,
         tokenizer=tokenizer,
         image_processor=image_processor),
    dict(type=RefCOCOGForGCGDataset,
         ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2014',
         json_file='data/RefCOCOg_GCG_train.json',
         local_path='data/coco/train2014',
         prompt_template=prompt_template,
         tokenizer=tokenizer,
         image_processor=image_processor),
    dict(type=FlickrForGCGDataset,
         ceph_path='BJ17:S3://wusize/flickr/train',
         json_file='data/flickr_mergedGT_GCG_train.json',
         local_path='data/flickr/train',
         prompt_template=prompt_template,
         tokenizer=tokenizer,
         image_processor=image_processor),
    # dict(type=PNGDataset,
    #      json_file='data/png_coco_train2017.json',
    #      panoptic_json_file='data/coco/annotations/panoptic_train2017.json',
    #      panoptic_png_path='data/coco/panoptic_train2017',
    #      tokenizer=tokenizer,
    #      image_processor=image_processor,
    #      prompt_template=prompt_template,
    #      local_path='data/coco/train2017',
    #      ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2017')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=dict(type=concat_datasets,
                 datasets_list=datasets_list),
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=gcg_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
