import os
from abc import ABC
from pathlib import Path
from collections import defaultdict
import pickle
import random
import numpy as np
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

from plyfile import PlyData
import lib.transforms as t
from lib.dataloader import InfSampler
from lib.voxelizer import Voxelizer


class DatasetPhase(Enum):
    Train = 0
    Val = 1
    Val2 = 2
    TrainVal = 3
    Test = 4


def datasetphase_2str(arg):
    if arg == DatasetPhase.Train:
        return 'train'
    elif arg == DatasetPhase.Val:
        return 'val'
    elif arg == DatasetPhase.Val2:
        return 'val2'
    elif arg == DatasetPhase.TrainVal:
        return 'trainval'
    elif arg == DatasetPhase.Test:
        return 'test'
    else:
        raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
    if arg.upper() == 'TRAIN':
        return DatasetPhase.Train
    elif arg.upper() == 'VAL':
        return DatasetPhase.Val
    elif arg.upper() == 'VAL2':
        return DatasetPhase.Val2
    elif arg.upper() == 'TRAINVAL':
        return DatasetPhase.TrainVal
    elif arg.upper() == 'TEST':
        return DatasetPhase.Test
    else:
        raise ValueError('phase must be one of train/val/test')


def cache(func):
    def wrapper(self, *args, **kwargs):
        # Assume that args[0] is index
        index = args[0]
        if self.cache:
            if index not in self.cache_dict[func.__name__]:
                results = func(self, *args, **kwargs)
                self.cache_dict[func.__name__][index] = results
            return self.cache_dict[func.__name__][index]
        else:
            return func(self, *args, **kwargs)

    return wrapper


class DictDataset(Dataset, ABC):
    IS_FULL_POINTCLOUD_EVAL = False

    def __init__(self,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 data_root='/'):
        """
        data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
        """
        Dataset.__init__(self)

        # Allows easier path concatenation
        if not isinstance(data_root, Path):
            data_root = Path(data_root)

        self.data_root = data_root
        self.data_paths = sorted(data_paths)

        self.prevoxel_transform = prevoxel_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

        # dictionary of input
        self.data_loader_dict = {
            'input': (self.load_input, self.input_transform),
            'target': (self.load_target, self.target_transform)
        }

        # For large dataset, do not cache
        self.cache = cache
        self.cache_dict = defaultdict(dict)
        self.loading_key_order = ['input', 'target']

    def load_input(self, index):
        raise NotImplementedError

    def load_target(self, index):
        raise NotImplementedError

    def get_classnames(self):
        pass

    def reorder_result(self, result):
        return result

    def __getitem__(self, index):
        out_array = []
        for k in self.loading_key_order:
            loader, transformer = self.data_loader_dict[k]
            v = loader(index)
            if transformer:
                v = transformer(v)
            out_array.append(v)
        return out_array

    def __len__(self):
        return len(self.data_paths)


class VoxelizationDatasetBase(DictDataset, ABC):
    IS_TEMPORAL = False
    CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
    ROTATION_AXIS = None
    NUM_IN_CHANNEL = None
    NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
    IGNORE_LABELS = None  # labels that are not evaluated

    def __init__(self,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 data_root='/',
                 ignore_mask=255,
                 return_transformation=False,
                 **kwargs):
        """
        ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
        """
        DictDataset.__init__(
            self,
            data_paths,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            cache=cache,
            data_root=data_root)

        self.ignore_mask = ignore_mask
        self.return_transformation = return_transformation

    def __getitem__(self, index):
        raise NotImplementedError

    def load_ply(self, index):
        filepath = self.data_root / self.data_paths[index]
        scene_name = self.data_paths[index]
        return self.load_ply_w_path(filepath, scene_name)

    def load_ply_w_path(self, filepath, scene_name):

        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)

        try:  # for scenes
            instances = np.array(data['instance_id'], dtype=np.int32)
        except:  # for sampled instances
            instances = None
            
        return coords, feats, labels, instances, scene_name

    def __len__(self):
        num_data = len(self.data_paths)
        return num_data


class VoxelizationDataset(VoxelizationDatasetBase):
    """This dataset loads RGB point clouds and their labels as a list of points
    and voxelizes the pointcloud with sufficient data augmentation.
    """
    # Voxelization arguments
    VOXEL_SIZE = 0.05  # 5cm

    # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
    # augmentation has to be done before voxelization
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
    ELASTIC_DISTORT_PARAMS = None

    # MISC.
    PREVOXELIZATION_VOXEL_SIZE = None

    # Augment coords to feats
    AUGMENT_COORDS_TO_FEATS = False

    def __init__(self,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 data_root='/',
                 ignore_label=255,
                 return_transformation=False,
                 augment_data=False,
                 config=None,
                 **kwargs):

        self.augment_data = augment_data
        self.config = config
        VoxelizationDatasetBase.__init__(
            self,
            data_paths,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            cache=cache,
            data_root=data_root,
            ignore_mask=ignore_label,
            return_transformation=return_transformation)

        # Prevoxel transformations
        self.voxelizer = Voxelizer(
            voxel_size=self.VOXEL_SIZE,
            clip_bound=self.CLIP_BOUND,
            use_augmentation=augment_data,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
            ignore_label=ignore_label)

        # map labels not evaluated to ignore_label
        # The valid labels are growing from 0-num_valid
        # Ignored label can be anything else
        # For one_hot calculations it is advised to set ignore label to num_valid
        # This way valid goes [0, num_valid-1], ignore [num_valid]
        label_map = {}
        inverse_label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                label_map[l] = self.ignore_mask
            else:
                label_map[l] = n_used
                inverse_label_map[n_used] = l
                n_used += 1
        label_map[self.ignore_mask] = self.ignore_mask
        inverse_label_map[self.ignore_mask] = 0
        self.label_map = label_map
        self.inverse_label_map = inverse_label_map

        self.NUM_LABELS -= len(self.IGNORE_LABELS)

    def _augment_coords_to_feats(self, coords, feats, labels=None):
        norm_coords = coords - coords.mean(0)
        # color must come first.
        if isinstance(coords, np.ndarray):
            feats = np.concatenate((feats, norm_coords), 1)
        else:
            feats = torch.cat((feats, norm_coords), 1)
        return coords, feats, labels

    def convert_mat2cfl(self, mat):
        # Generally, xyz,rgb,label
        return mat[:, :3], mat[:, 3:-1], mat[:, -1]

    def __getitem__(self, index):

        coords, feats, labels, instance_ids, scene_name = self.load_ply(index)

        if self.config.sample_tail_instances and self.augment_data:
            coords, feats, labels = add_instances_to_cloud(coords, feats, labels, scene_name,
                                                           instance_weights=self.instance_sampling_weights,
                                                           config=self.config,
                                                           valid_labels=self.valid_labels)

        # Downsample the pointcloud with finer voxel size before transformation for memory and speed
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            _, inds = ME.utils.sparse_quantize(
                coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
            coords = coords[inds]
            feats = feats[inds]
            labels = labels[inds]

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords, feats, labels = self.prevoxel_transform(coords, feats, labels)

        coords, feats, labels, transformation = self.voxelizer.voxelize(
            coords, feats, labels)

        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, feats, labels = self.input_transform(coords, feats, labels)
        if self.target_transform is not None:
            coords, feats, labels = self.target_transform(coords, feats, labels)
        if self.IGNORE_LABELS is not None:
            # labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

            mapper = lambda x: self.label_map[x]
            labels = np.vectorize(mapper)(labels)


        # Use coordinate features if config is set
        if self.AUGMENT_COORDS_TO_FEATS:
            coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)

        return_args = [coords, feats, labels]
        if self.return_transformation:
            return_args.append(transformation.astype(np.float32))

        return_args.append(scene_name)

        return tuple(return_args)

def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           collate_function=None,
                           input_transform=None,
                           target_transform=None):
    if isinstance(phase, str):
        phase = str2datasetphase_type(phase)

    if config.return_transformation:
        collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
    else:
        if collate_function is None:
            collate_fn = t.cfl_collate_fn_factory(limit_numpoints)
        else:
            collate_fn = collate_function(limit_numpoints)

    prevoxel_transform_train = []
    if augment_data and config.elastic_distortion:
        prevoxel_transform_train.append(t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS))

    if len(prevoxel_transform_train) > 0:
        prevoxel_transforms = t.Compose(prevoxel_transform_train)
    else:
        prevoxel_transforms = None

    input_transforms = []
    if input_transform is not None:
        input_transforms += input_transform

    if augment_data:
        input_transforms += [
            t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
            # t.ChromaticAutoContrast(),
            # t.ChromaticTranslation(config.data_aug_color_trans_ratio),
            # t.ChromaticJitter(config.data_aug_color_jitter_std),
            # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
        ]

    # if config.data_aug_color_scaling_factor != 1.0:
    #     input_transforms += [t.ChromaticScale(scale_factor=config.data_aug_color_scaling_factor)]

    if config.data_aug_patch_dropout_ratio == 0.:
        input_transforms += [t.RandomDropout(0.2)]

    if len(input_transforms) > 0:
        input_transforms = t.Compose(input_transforms)
    else:
        input_transforms = None

    dataset = DatasetClass(
        config,
        prevoxel_transform=prevoxel_transforms,
        input_transform=input_transforms,
        target_transform=target_transform,
        cache=config.cache_data,
        augment_data=augment_data,
        phase=phase)

    data_args = {
        'dataset': dataset,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'collate_fn': collate_fn,
    }

    if repeat:
        data_args['sampler'] = InfSampler(dataset, shuffle)
    else:
        data_args['shuffle'] = shuffle

    data_loader = DataLoader(**data_args)

    return data_loader
