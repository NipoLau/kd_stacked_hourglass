import argparse
from config.default import get_cfg
from models import teacher_hourglass, student_hourglass
from torch.nn import DataParallel
from core.loss import JointsMSELoss
from dataset.mpii import MPIIDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from core.function import kd_train, hint_train, train, kd_train_inter
from torch.nn import Conv2d, MSELoss
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import os
from collections import OrderedDict
from core.function import evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        help='model configuration file',
        required=True,
        type=str
    )

    parser.add_argument(
        '--tcfg',
        help='teacher model configuration file',
        required=False,
        default=None,
        type=str
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_cfg(args.cfg)
    model = eval(cfg.MODEL.NAME + '.get_pose_net')(cfg)
    model = DataParallel(model.cuda())

    if cfg.MODEL.PRETRAINED:
        state_dict = torch.load(cfg.MODEL.PRETRAINED)
        model.load_state_dict(state_dict)

    pose_criterion = JointsMSELoss().cuda()

    train_type = cfg.TRAIN.TYPE

    regressor = None

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if train_type == 'KD':
        tcfg = get_cfg(args.tcfg)
        tmodel = eval(tcfg.MODEL.NAME + '.get_pose_net')(tcfg)
        tmodel = DataParallel(tmodel.cuda())

        state_dict = torch.load(tcfg.MODEL.PRETRAINED)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v
        tmodel.load_state_dict(new_state_dict, strict=True)

        kd_pose_criterion = JointsMSELoss().cuda()
        hint_criterion = MSELoss(reduction='mean').cuda()

        train_type = 'KD'

        regressor = Conv2d(cfg.MODEL.EXTRA.NUM_FEATURES, tcfg.MODEL.EXTRA.NUM_FEATURES,
                           kernel_size=1, stride=1).cuda()

    train_dataset = MPIIDataset(
        cfg, cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

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

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=cfg.TRAIN.LR
    )

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR
    )

    name_values = evaluate(cfg, valid_loader, valid_dataset, model)

    print('---------Student Model----------')
    best_perf = name_values['Mean']

    for k, v in name_values.items():
        print('{} : {}'.format(k, v))
    '''
    name_values = evaluate(tcfg, valid_loader, valid_dataset, tmodel)

    print('---------Teacher Model----------')

    for k, v in name_values.items():
        print('{} : {}'.format(k, v))
    '''

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        if train_type == 'KD':
            if epoch < cfg.TRAIN.HINT_EPOCH:
                hint_train(cfg, model, tmodel, regressor, train_loader, hint_criterion, optimizer, epoch)
            else:
                if cfg.TRAIN.INTER:
                    kd_train_inter(cfg, model, tmodel, train_loader, pose_criterion, kd_pose_criterion, optimizer, epoch)
                else:
                    kd_train(cfg, model, tmodel, train_loader, pose_criterion, kd_pose_criterion, optimizer, epoch)
        else:
            train(cfg, model, train_loader, pose_criterion, optimizer, epoch)

        name_values = evaluate(cfg, valid_loader, valid_dataset, model)

        for k, v in name_values.items():
            print('{} : {}'.format(k, v))

        if name_values['Mean'] > best_perf:
            best_perf = name_values['Mean']
            torch.save(model.state_dict(), os.path.join(cfg.TRAIN.OUTPUT, 'checkpoint.pth'))


if __name__ == '__main__':
    main()
