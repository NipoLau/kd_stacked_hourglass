import cv2
import numpy as np


def get_affine_transform(center, scale, rotate, output_size, inv=0):
    scale = scale * 200

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rotate_rad = np.pi * rotate / 180

    src_dir = get_dir([0, src_w * -0.5], rotate_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0] = center
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    src[1] = src[0] + src_dir
    dst[1] = dst[0] + dst_dir
    src[2] = get_3rd_point(src[0], src[1])
    dst[2] = get_3rd_point(dst[0], dst[1])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(coord, trans):
    coord = np.array([coord[0], coord[1], 1.]).T
    coord = np.dot(trans, coord)
    return coord[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rotate_rad):
    sn, cs = np.sin(rotate_rad), np.cos(rotate_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
