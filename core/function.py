import torch
import numpy as np
import prettytable as pt
import time
from utils.utils import get_final_preds, get_max_preds, nms


def accuracy(output, target):
    '''
    计算 PCK 精度，输出坐标与目标坐标之间的距离除以热图宽高
    '''
    pred = get_max_preds(output).astype(np.float32)
    target = get_max_preds(target).astype(np.float32)
    threshold = 0.5

    batch_size = output.shape[0]
    num_joints = output.shape[1]
    h = output.shape[2]
    w = output.shape[3]
    norm = np.ones((batch_size, 2)) * np.array([h, w]) / 10

    dists = np.zeros((num_joints, batch_size))
    for i in range(batch_size):
        for j in range(num_joints):
            if target[i, j, 0] > 1 and target[i, j, 1] > 1:
                norm_pred = pred[i, j] / norm[i]
                norm_target = target[i, j] / norm[i]
                dists[j, i] = np.linalg.norm(norm_pred - norm_target)
            else:
                dists[j, i] = -1

    acc = np.zeros(num_joints)

    for i in range(num_joints):
        dist_cal = np.not_equal(dists[i], -1)
        if dist_cal.sum() > 0:
            acc[i] = np.less(dists[i][dist_cal], threshold).sum() * 1.0 / dist_cal.sum()
        else:
            acc[i] = -1

    return acc


def print_table(outputs, target, loss):
    acc = accuracy(outputs[-1].detach().cpu().numpy(),
                   target.detach().cpu().numpy())

    head = 9
    lsho, rsho = 13, 12
    lelb, relb = 14, 11
    lwri, rwri = 15, 10
    lhip, rhip = 3, 2
    lkne, rkne = 4, 1
    lank, rank = 5, 0

    name_value = [
        ('Head', acc[head]),
        ('Shoulder', 0.5 * (acc[lsho] + acc[rsho])),
        ('Elbow', 0.5 * (acc[lelb] + acc[relb])),
        ('Wrist', 0.5 * (acc[lwri] + acc[rwri])),
        ('Hip', 0.5 * (acc[lhip] + acc[rhip])),
        ('Knee', 0.5 * (acc[lkne] + acc[rkne])),
        ('Ankle', 0.5 * (acc[lank] + acc[rank])),
        ('Loss', loss.item())
    ]

    table = pt.PrettyTable()
    table.field_names = [name_value[i][0] for i in range(len(name_value))]
    table.add_row([name_value[i][1] for i in range(len(name_value))])
    table.float_format = '.6'

    print(table)


def train(config, model, train_loader, pose_criterion, optimizer, epoch):
    model.train()

    print('--------' + 'EPOCH:' + str(epoch) + '--------')

    for i, (input, target, target_vis, meta) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        target_vis = target_vis.cuda()
        _, outputs = model(input)

        pose_loss = pose_criterion(outputs[0], target, target_vis)

        for idx in range(1, len(outputs), 1):
            pose_loss += pose_criterion(outputs[idx], target, target_vis)

        loss = pose_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.PRINT_FREQ == 0:
            print_table(outputs, target, loss)


def kd_train(config, model, tmodel, train_loader, pose_criterion, kd_pose_criterion, optimizer, epoch):
    model.train()
    tmodel.eval()

    print('--------' + 'KD:' + str(epoch) + '--------')

    for i, (input, target, target_vis, meta) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        target_vis = target_vis.cuda()
        _, outputs = model(input)

        with torch.no_grad():
            _, toutputs = tmodel(input)

        pose_loss = pose_criterion(outputs[0], target, target_vis)
        kd_pose_loss = kd_pose_criterion(outputs[0], toutputs[-1], target_vis)

        for idx in range(1, len(outputs), 1):
            pose_loss += pose_criterion(outputs[idx], target, target_vis)
            kd_pose_loss += kd_pose_criterion(outputs[idx], toutputs[-1], target_vis)

        loss = config.TRAIN.KD_WEIGHT * kd_pose_loss + (1 - config.TRAIN.KD_WEIGHT) * pose_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.PRINT_FREQ == 0:
            print_table(outputs, target, loss)


def kd_train_inter(config, model, tmodel, train_loader, pose_criterion, kd_pose_criterion, optimizer, epoch):
    model.train()
    tmodel.eval()

    print('--------' + 'INTER:' + str(epoch) + '--------')

    for i, (input, target, target_vis, meta) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        target_vis = target_vis.cuda()
        _, outputs = model(input)

        with torch.no_grad():
            _, toutputs = tmodel(input)

        pose_loss = pose_criterion(outputs[0], target, target_vis)
        kd_pose_loss = kd_pose_criterion(outputs[0], toutputs[1], target_vis)

        for idx in range(1, len(outputs), 1):
            pose_loss += pose_criterion(outputs[idx], target, target_vis)
            kd_pose_loss += kd_pose_criterion(outputs[idx], toutputs[2 * idx + 1], target_vis)

        loss = config.TRAIN.KD_WEIGHT * kd_pose_loss + (1 - config.TRAIN.KD_WEIGHT) * pose_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.PRINT_FREQ == 0:
            print_table(outputs, target, loss)


def hint_train(config, model, tmodel, regressor, train_loader, hint_criterion, optimizer, epoch):
    model.train()
    tmodel.eval()

    print('--------' + 'HINT:' + str(epoch) + '--------')

    for i, (input, target, target_vis, meta) in enumerate(train_loader):
        input = input.cuda()
        hint, outputs = model(input)
        with torch.no_grad():
            thint, _ = tmodel(input)

        out_hint = regressor(hint)

        loss = hint_criterion(out_hint, thint)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.PRINT_FREQ == 0:
            print_table(outputs, target, loss)


def evaluate(config, val_loader, val_dataset, model):
    model.eval()

    device = torch.device(config.MODEL.DEVICE + ":" + config.MODEL.DEVICE_ID)
    idx = 0

    with torch.no_grad():
        all_preds = np.zeros((len(val_dataset), config.MODEL.NUM_JOINTS, 2))

        end = time.time()

        for i, (input, target, target_vis, meta) in enumerate(val_loader):
            input = input.to(device)
            input_flipped = torch.flip(input, dims=(3,))
            num_images = len(input)
            _, outputs = model(input)
            _, outputs_flipped = model(input_flipped)
            if isinstance(outputs, list):
                output = outputs[-1]
                output_flipped = outputs_flipped[-1]

            output_flipped = torch.flip(output_flipped, (3,))

            output = (output + output_flipped[:, val_dataset.flipped_joints]) / 2

            output = nms(output)

            all_preds[idx:idx + num_images] = get_final_preds(output.cpu().numpy(), meta['center'],
                                                              meta['scale'], config.MODEL.HEATMAP_SIZE)
            idx += num_images

        name_values = val_dataset.evaluate(
            config, all_preds
        )

        print(time.time() - end)

    return name_values
