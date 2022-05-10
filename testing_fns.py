# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import copy
import os.path as osp
import time
import torch

import mmcv
import torch
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.datasets import (build_dataloader, replace_ImageToTensor)
from mmdet.core import DistEvalHook, EvalHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import collect_env, get_root_logger, setup_multi_processes, compat_cfg, find_latest_checkpoint, get_root_logger

def setup(config: str,
          seed: Optional[int] = 42,
          deterministic: bool = True,
          ):
    # load cfg
    cfg = mmcv.Config.fromfile(config)
    # set multi-process settings
    setup_multi_processes(cfg)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)
    # setup work_dir
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
    # setup gpus
    cfg.gpu_ids = range(1)
    # distributed
    distributed = False
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    # set random seeds
    seed = init_random_seed(seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {deterministic}')
    set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed # type: ignore
    meta['exp_name'] = osp.basename(config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    return model, datasets, cfg, timestamp, meta

def load_image(datasets, i):
    data = datasets[0][i]
    return dict(img=data['img'].data[None], 
                img_metas=[data['img_metas'].data], 
                gt_bboxes=[data['gt_bboxes'].data], 
                gt_labels=[data['gt_labels'].data]) 
