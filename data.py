import copy
import logging
from functools import partial
from typing import Dict, Optional, Union, List

from mmengine.runner import Runner
from mmengine.evaluator import Evaluator
from mmengine.dataset import worker_init_fn
from mmengine.dist import get_rank
from mmengine.logging import print_log
from mmengine.registry import DATA_SAMPLERS, FUNCTIONS, EVALUATOR, VISUALIZERS
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


from torch.utils.data import DataLoader

from mmrotate.registry import DATASETS


def build_data_loader(data_name=None):
    if data_name is None or data_name == 'trainval_with_hbox':
        return MMEngine_build_dataloader(dataloader=naive_trainval_dataloader)
    elif data_name == 'test_without_hbox':
        return MMEngine_build_dataloader(dataloader=naive_test_dataloader)
    else:
        raise NotImplementedError()


def build_evaluator(merge_patches=True, format_only=False):
    naive_evaluator.update(dict(
        merge_patches=merge_patches, format_only=format_only))
    return MMEngine_build_evaluator(evaluator=naive_evaluator)


def build_visualizer():
    vis_backends = [dict(type='LocalVisBackend')]
    visualizer = dict(
        type='RotLocalVisualizer', vis_backends=vis_backends,
        name='sammrotate', save_dir='./rbbox_vis')
    return VISUALIZERS.build(visualizer)


# dataset settings
dataset_type = 'DOTADataset'
data_root = 'data/split_ss_dota/'
backend_args = None

naive_trainval_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='AddConvertedGTBox', box_type_mapping=dict(h_gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # # Horizontal GTBox, (x,y,w,h,theta)
    # dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'h_gt_bboxes'))
]

naive_test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

naive_trainval_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='trainval/annfiles/',
    # ann_file='trainval/annfiles-1sample/',
    # ann_file='trainval/annfiles-3sample/',
    # ann_file='trainval/annfiles-10sample/',
    # ann_file='trainval/annfiles-30sample/',
    # ann_file='trainval/annfiles-100sample/',
    ann_file='trainval/annfiles-1000sample/',
    data_prefix=dict(img_path='trainval/images/'),
    test_mode=True,  # we only inference the sam
    pipeline=naive_trainval_pipeline)

naive_test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='test/images/'),
    test_mode=True,
    pipeline=naive_test_pipeline)

naive_trainval_dataloader = dict(
    batch_size=1,
    # num_workers=0,  # For debug
    num_workers=2,
    # persistent_workers=False,  # For debug
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=naive_trainval_dataset)

naive_test_dataloader = dict(
    batch_size=1,
    # num_workers=0,  # For debug
    num_workers=2,
    # persistent_workers=False,  # For debug
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=naive_test_dataset)

naive_evaluator = dict(
    type='DOTAMetric', metric='mAP', outfile_prefix='./work_dirs/dota/Task1')


def MMEngine_build_dataloader(dataloader: Union[DataLoader, Dict],
                              seed: Optional[int] = None,
                              diff_rank_seed: bool = False) -> DataLoader:
    """Build dataloader.

    The method builds three components:

    - Dataset
    - Sampler
    - Dataloader

    An example of ``dataloader``::

        dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=9
        )

    Args:
        dataloader (DataLoader or dict): A Dataloader object or a dict to
            build Dataloader object. If ``dataloader`` is a Dataloader
            object, just returns itself.
        seed (int, optional): Random seed. Defaults to None.
        diff_rank_seed (bool): Whether or not set different seeds to
            different ranks. If True, the seed passed to sampler is set
            to None, in order to synchronize the seeds used in samplers
            across different ranks.


    Returns:
        Dataloader: DataLoader build from ``dataloader_cfg``.
    """
    if isinstance(dataloader, DataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop('dataset')
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
    else:
        # fallback to raise error in dataloader
        # if `dataset_cfg` is not a valid type
        dataset = dataset_cfg

    # build sampler
    sampler_cfg = dataloader_cfg.pop('sampler')
    if isinstance(sampler_cfg, dict):
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg,
            default_args=dict(dataset=dataset, seed=sampler_seed))
    else:
        # fallback to raise error in dataloader
        # if `sampler_cfg` is not a valid type
        sampler = sampler_cfg

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler,
                batch_size=dataloader_cfg.pop('batch_size')))
    else:
        # fallback to raise error in dataloader
        # if `batch_sampler_cfg` is not a valid type
        batch_sampler = batch_sampler_cfg

    # build dataloader
    init_fn: Optional[partial]

    if seed is not None:
        disable_subprocess_warning = dataloader_cfg.pop(
            'disable_subprocess_warning', False)
        assert isinstance(
            disable_subprocess_warning,
            bool), ('disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
        init_fn = partial(
            worker_init_fn,
            num_workers=dataloader_cfg.get('num_workers'),
            rank=get_rank(),
            seed=seed,
            disable_subprocess_warning=disable_subprocess_warning)
    else:
        init_fn = None

    # `persistent_workers` requires pytorch version >= 1.7
    if ('persistent_workers' in dataloader_cfg
            and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
        print_log(
            '`persistent_workers` is only available when '
            'pytorch version >= 1.7',
            logger='current',
            level=logging.WARNING)
        dataloader_cfg.pop('persistent_workers')

    # The default behavior of `collat_fn` in dataloader is to
    # merge a list of samples to form a mini-batch of Tensor(s).
    # However, in mmengine, if `collate_fn` is not defined in
    # dataloader_cfg, `pseudo_collate` will only convert the list of
    # samples into a dict without stacking the batch tensor.
    collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                        dict(type='pseudo_collate'))
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = FUNCTIONS.get(collate_fn_type)
    collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        **dataloader_cfg)
    return data_loader


def MMEngine_build_evaluator(evaluator: Union[Dict, List, Evaluator]) -> Evaluator:
    """Build evaluator.

    Examples of ``evaluator``::

        # evaluator could be a built Evaluator instance
        evaluator = Evaluator(metrics=[ToyMetric()])

        # evaluator can also be a list of dict
        evaluator = [
            dict(type='ToyMetric1'),
            dict(type='ToyEvaluator2')
        ]

        # evaluator can also be a list of built metric
        evaluator = [ToyMetric1(), ToyMetric2()]

        # evaluator can also be a dict with key metrics
        evaluator = dict(metrics=ToyMetric())
        # metric is a list
        evaluator = dict(metrics=[ToyMetric()])

    Args:
        evaluator (Evaluator or dict or list): An Evaluator object or a
            config dict or list of config dict used to build an Evaluator.

    Returns:
        Evaluator: Evaluator build from ``evaluator``.
    """
    if isinstance(evaluator, Evaluator):
        return evaluator
    elif isinstance(evaluator, dict):
        # if `metrics` in dict keys, it means to build customized evalutor
        if 'metrics' in evaluator:
            evaluator.setdefault('type', 'Evaluator')
            return EVALUATOR.build(evaluator)
        # otherwise, default evalutor will be built
        else:
            return Evaluator(evaluator)  # type: ignore
    elif isinstance(evaluator, list):
        # use the default `Evaluator`
        return Evaluator(evaluator)  # type: ignore
    else:
        raise TypeError(
            'evaluator should be one of dict, list of dict, and Evaluator'
            f', but got {evaluator}')


