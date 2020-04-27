import numpy as np
import torch
from utils.transforms import get_affine_transform, affine_transform


def adjust(ans, det):
    for batch_id, people in enumerate(ans):
        for people_id, i in enumerate(people):
            for joint_id, joint in enumerate(i):
                if joint[0] != 0:
                    y, x = joint[0:2]
                    xx, yy = int(x), int(y)
                    tmp = det[0][joint_id]
                    if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                        y += 0.25
                    else:
                        y -= 0.25

                    if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                        x += 0.25
                    else:
                        x -= 0.25
                    ans[batch_id][people_id, joint_id, 0:2] = (y + 0.5, x + 0.5)
    return ans


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[2]

    batch_heatmaps = batch_heatmaps.reshape((batch_size, num_joints, -1))

    idx = np.argmax(batch_heatmaps, 2).reshape((batch_size, num_joints, 1))
    val = np.amax(batch_heatmaps, 2).reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = np.floor(preds[:, :, 1] / width)

    preds_mask = np.tile(np.greater(val, 0.0), (1, 1, 2)).astype(np.float32)
    preds *= preds_mask

    return preds


def get_final_preds(batch_heatmaps, center, scale, output_size):
    coords = get_max_preds(batch_heatmaps)
    coords = adjust([coords], batch_heatmaps)[0]

    for i in range(len(coords)):
        trans = get_affine_transform(center[i], scale[i], 0, output_size, inv=1)
        for j in range(len(coords[i])):
            coords[i][j] = affine_transform(coords[i][j], trans)

    return coords


def generate_target(joints, joints_vis, heatmap_size, ratio, sigma):
    num_joints = len(joints)

    target = np.zeros((num_joints,
                       heatmap_size[0],
                       heatmap_size[1]),
                      dtype=np.float32)
    target_vis = joints_vis

    for joint_id in range(num_joints):
        mu_x = int(joints[joint_id][0] / ratio + 0.5)
        mu_y = int(joints[joint_id][1] / ratio + 0.5)

        tmp_size = 3 * sigma

        ul = np.array([int(mu_x - tmp_size), int(mu_y - tmp_size)])
        # numpy 索引为 左闭右开，二位高斯分布直径为 2 * tmp_size + 1
        br = np.array([int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)])
        # 部分关节坐标会偏出由 center + scale 划定的矩形区域内，则将该关节坐标视为不可见
        # 去除无效坐标点，否则生成热图会出错
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            target_vis[joint_id] = 0

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]

        x0 = y0 = size // 2

        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # img_x 和 img_y 对应生成热图的二维高斯分布区域的四角坐标
        img_x = np.array([max(0, ul[0]), min(heatmap_size[0], br[0])])
        img_y = np.array([max(0, ul[1]), min(heatmap_size[1], br[1])])

        # g_x 和 g_y 对应 g 的四角坐标
        g_x, g_y = img_x - ul[0], img_y - ul[1]

        if target_vis[joint_id] != 0:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_vis


def nms(output):
    pool = torch.nn.MaxPool2d(3, 1, 1)

    maxm = pool(output)
    maxm = torch.eq(maxm, output).float()

    output = output * maxm

    return output
