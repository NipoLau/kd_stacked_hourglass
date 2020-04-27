from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import h5py
import random
from utils.transforms import get_affine_transform, affine_transform
from utils.utils import generate_target
from collections import OrderedDict


class MPIIDataset(Dataset):
    def __init__(self, cfg, root, image_set, transform):
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.db = self._get_db()
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.flipped_joints = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
        self.matched_joints = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR

    def _get_db(self):
        db_file = os.path.join(self.root, 'annot', self.image_set + '.h5')
        gt_db = []

        anno = h5py.File(db_file, 'r')

        image_name = anno['imgname'][()]
        center = anno['center'][()]
        scale = anno['scale'][()]
        joints = anno['part'][()]
        joints_vis = anno['visible'][()]
        normalize = anno['normalize'][()]

        for idx in range(len(image_name)):
            gt_db.append(
                {
                    'image': os.path.join(self.root, 'images', image_name[idx].decode('UTF-8')),
                    'center': center[idx],
                    'scale': np.array([scale[idx], scale[idx]]),
                    'joints': joints[idx],
                    'joints_vis': joints_vis[idx],
                    'normalize': normalize[idx]
                }
            )

        return gt_db

    def __len__(self):
        return len(self.db)

    # image, target_heatmap, target_visible
    def __getitem__(self, idx):
        db_rec = self.db[idx]

        image_file = db_rec['image']
        image_numpy = cv2.imread(image_file)[:, :, ::-1]

        joints = db_rec['joints']
        joints_vis = db_rec['joints_vis']

        center = db_rec['center']
        scale = db_rec['scale']
        rotate = 0

        if self.image_set == 'train':
            sf = self.scale_factor
            rf = self.rot_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotate = np.clip(np.random.randn() * rf, - rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                image_numpy = np.flip(image_numpy, axis=1)
                joints[:, 0] = image_numpy.shape[1] - joints[:, 0] - 1
                for joint in self.matched_joints:
                    joints[joint[0]], joints[joint[1]] = \
                        joints[joint[1]], joints[joint[0]].copy()
                    joints_vis[joint[0]], joints_vis[joint[1]] = \
                        joints_vis[joint[1]], joints_vis[joint[0]].copy()
                center[0] = image_numpy.shape[1] - center[0] - 1

        trans = get_affine_transform(center, scale, rotate, self.image_size)

        image = cv2.warpAffine(
            image_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            image = self.transform(image)

        # 将关节坐标转换到输入图像（256x256）中坐标
        for i in range(self.num_joints):
            if joints_vis[i] != 0:
                joints[i] = affine_transform(joints[i], trans)

        target, target_vis = generate_target(
            joints,
            joints_vis,
            self.heatmap_size,
            self.image_size[0] / self.heatmap_size[0],
            self.sigma)

        meta = {
            'center': center,
            'scale': scale
        }

        return image, target, target_vis, meta

    def evaluate(self, cfg, preds, threshold=0.5):

        pos_gt_src = np.array([self.db[idx]['joints'] for idx in range(self.__len__())])
        normalize = np.array([self.db[idx]['normalize'] for idx in range(self.__len__())])
        jnt_visible = np.array(pos_gt_src[:, :, 0] != 0)

        uv_error = preds - pos_gt_src
        uv_error = np.linalg.norm(uv_error, axis=2)

        scale = np.tile(normalize[:, np.newaxis], (1, 16))
        scaled_uv_error = np.divide(uv_error, scale)
        scaled_uv_error = np.multiply(scaled_uv_error, jnt_visible)

        jnt_count = np.sum(jnt_visible, axis=0)

        less_than_threshold = np.multiply(
            (scaled_uv_error <= threshold), jnt_visible
        )

        PCKh = np.divide(100. * np.sum(less_than_threshold, axis=0), jnt_count)

        jnt_ratio = np.divide(jnt_count, np.sum(jnt_count).astype(np.float64))

        name_value = [
            ('Head', PCKh[9]),
            ('Shoulder', 0.5 * (PCKh[12] + PCKh[13])),
            ('Elbow', 0.5 * (PCKh[11] + PCKh[14])),
            ('Wrist', 0.5 * (PCKh[10] + PCKh[15])),
            ('Hip', 0.5 * (PCKh[3] + PCKh[2])),
            ('Knee', 0.5 * (PCKh[4] + PCKh[1])),
            ('Ankle', 0.5 * (PCKh[0] + PCKh[5])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
        ]

        return OrderedDict(name_value)
