import glob
import logging
import os
import random
import sys
from pathlib import Path
import pickle

import numpy as np
from plyfile import PlyData
from scipy import spatial, ndimage, misc
import torch

from lib.constants.dataset_sets import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.datasets.front3d import Front3DLightingGeometryDataset, Front3DLightGeometry2cmDataset
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.transforms import InstanceAugmentation
from lib.utils import read_txt, fast_hist, per_class_iu

from lib.constants.scannet_constants import *
from lib.datasets.preprocessing.utils import box_intersect

import MinkowskiEngine as ME


class LampSemanticsDataset(Front3DLightGeometry2cmDataset):
    def collect_category_weights(self):
        return None

    def load_sample(self, index):
        scene_name = self.data_paths[index]
        coords_filepath = os.path.join(self.data_root, scene_name, self.COORDS_FILE_PATH)
        coords = np.load(coords_filepath)

        return coords, np.zeros_like(coords) + 0.5, np.zeros(len(coords)), scene_name


class Front3DLightGeometry2cmDataset(Front3DLightingGeometryDataset):
    VOXEL_SIZE = 0.02



