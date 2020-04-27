import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target, target_vis):
        batch_size = output.shape[0]
        num_joints = output.shape[1]

        heatmaps_pred = output.reshape(batch_size, num_joints, -1)
        heatmaps_gt = target.reshape(batch_size, num_joints, -1)

        loss = 0.0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[:, idx]
            heatmap_gt = heatmaps_gt[:, idx]

            loss += 0.5 * self.criterion(
                heatmap_pred.mul(target_vis[:, idx].unsqueeze(1)),
                heatmap_gt.mul(target_vis[:, idx].unsqueeze(1))
            )

        return loss / num_joints
