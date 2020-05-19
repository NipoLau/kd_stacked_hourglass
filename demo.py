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
    model = DataParallel(model.cuda())

    # load pre-trained teacher network
    state_dict = torch.load(cfg.MODEL.PRETRAINED)
    if isinstance(state_dict, OrderedDict):
        new_state_dict = state_dict
    else:
        state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v
    model.load_state_dict(new_state_dict, strict=True)

    valid_dataset = MPIIDataset(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        transforms.ToTensor()
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    device = torch.device(cfg.MODEL.DEVICE + ":" + cfg.MODEL.DEVICE_ID)

    for i, (input, target, target_vis, meta) in enumerate(valid_loader):
        input = input.to(device)
        _, output = model(input)
        input = input.cpu().numpy()
        output = output.cpu().numpy()

        plt.figure()

        for k in range(len(output)):
            ax = plt.subplot(4, 6, k + 1)
            ax.imshow(input[k])

if __name__ == '__main__':
    main()
