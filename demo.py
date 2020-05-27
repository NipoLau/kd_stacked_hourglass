from config.default import get_cfg
import torch
from models import teacher_hourglass, student_hourglass
from collections import OrderedDict
from torchvision.transforms import transforms
from dataset.mpii import MPIIDataset
from core.function import evaluate
from torch.nn import DataParallel
from utils.utils import get_max_preds
import torch.utils.data
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.onnx as onnx

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        help='configuration file path',
        required=True,
        type=str
    )

    return parser.parse_args()


def main():
    args = parse_arg()
    cfg = get_cfg(args.cfg)

    model = eval(cfg.MODEL.NAME + '.get_pose_net')(cfg)
    # model = DataParallel(model.cuda())

    # load pre-trained teacher network
    state_dict = torch.load(cfg.MODEL.PRETRAINED)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    input = torch.randn(1, 3, 256, 256)

    input_names = ["input"]
    output_names = ["output"]

    onnx.export(
        model,
        input,
        "experiments/output/kd_hourglass.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names
    )

    end = time.time()
    output = model(input)
    print(time.time() - end)


if __name__ == '__main__':
    main()
