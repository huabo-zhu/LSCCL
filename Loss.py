import torch
import numpy as np
from setting import *


class torch_loss_centroid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, location_pred, location_true):

        ctx.save_for_backward(location_pred, location_true)

        loss = 0
        for batch_idx in range(location_pred.size()[0]):
            x_true, y_true = location_true[batch_idx, :, 0], location_true[batch_idx, :, 1]
            x_pred, y_pred = location_pred[batch_idx, :, 0], location_pred[batch_idx, :, 1]
            len_true, len_pred = x_true.size()[0], x_pred.size()[0]

            '''
            np.square : 元素平方
            x_true size为[37]
            x_true.unsqueeze(0).expand(len_true, -1)及其转置的size为[37,37]
            '''
            # x_true为一幅图中所有点的x坐标，不齐37个点则由0补齐，是一个一维向量，shape[37]
            # x_true.unsqueeze(0) 0轴增加一个维度，shape[1,37]
            # x_true.unsqueeze(0).expand(len_true, -1) 将shape扩展到[37,37]，即37行数据，每行都是重复的x_true
            # x_true.unsqueeze(0).expand(len_true, -1).transpose(0, 1)是转置，即每列都是重复的x_true，37列
            term1 = np.pi * np.square(smoothing_lambda) / 2 * torch.exp(
                - ((x_true.unsqueeze(0).expand(len_true, -1) -
                    x_true.unsqueeze(0).expand(len_true, -1).transpose(0, 1)) ** 2
                   + (y_true.unsqueeze(0).expand(len_true, -1) - y_true.unsqueeze(0).expand(len_true, -1).transpose(
                            0, 1)) ** 2) / (2 * np.square(smoothing_lambda)))

            term2 = np.pi * np.square(smoothing_lambda) / 2 * torch.exp(
                - ((x_pred.unsqueeze(0).expand(len_pred, -1) - x_pred.unsqueeze(0).expand(len_pred, -1).transpose(0,
                                                                                                                  1)) ** 2
                   + (y_pred.unsqueeze(0).expand(len_pred, -1) - y_pred.unsqueeze(0).expand(len_pred, -1).transpose(
                            0,
                            1)) ** 2)
                / (2 * np.square(smoothing_lambda)))

            term3 = np.pi * np.square(smoothing_lambda) / 2 * torch.exp(
                - ((x_pred.unsqueeze(0).expand(len_true, -1) - x_true.unsqueeze(0).expand(len_pred, -1).transpose(0,
                                                                                                                  1)) ** 2
                   + (y_pred.unsqueeze(0).expand(len_true, -1) - y_true.unsqueeze(0).expand(len_pred, -1).transpose(
                            0,
                            1)) ** 2)
                / (2 * np.square(smoothing_lambda)))

            for channel_idx in range(location_pred.size()[2] - 2):
                p_pred = location_pred[batch_idx, :, 2 + channel_idx]
                p_true = location_true[batch_idx, :, 2 + channel_idx]

                loss += torch.sum(p_true.unsqueeze(0) * p_true.unsqueeze(0).transpose(0, 1) * term1) \
                        + torch.sum(p_pred.unsqueeze(0) * p_pred.unsqueeze(0).transpose(0, 1) * term2) \
                        - 2 * torch.sum(
                    p_pred.unsqueeze(0).expand(len_true, -1) * p_true.unsqueeze(0).expand(len_pred, -1).transpose(0,
                                                                                                                  1) * term3)
                te1 = torch.sum(p_true.unsqueeze(0) * p_true.unsqueeze(0).transpose(0, 1) * term1)
                te2 = torch.sum(p_pred.unsqueeze(0) * p_pred.unsqueeze(0).transpose(0, 1) * term2)
                te3 = - 2 * torch.sum(
                    p_pred.unsqueeze(0).expand(len_true, -1) * p_true.unsqueeze(0).expand(len_pred, -1).transpose(0,
                                                                                                                  1) * term3)
        #         print("*"*100)
        #         print("te1: ", te1)
        #         print("te2: ", te2)
        #         print("te3: ", te3)
        # print("loss: ", loss)
        return loss

    @staticmethod
    def backward(ctx, grad):

        location_pred, location_true, = ctx.saved_tensors

        FullGradient = []
        for batch_idx in range(location_pred.size()[0]):

            x_pred, y_pred = location_pred[batch_idx, :, 0], location_pred[batch_idx, :, 1]
            x_true, y_true = location_true[batch_idx, :, 0], location_true[batch_idx, :, 1]

            len_true, len_pred = x_true.size()[0], x_pred.size()[0]

            x_dist_pred_pred, y_dist_pred_pred = \
                x_pred.unsqueeze(0).expand(len_pred, -1) - x_pred.unsqueeze(0).expand(len_pred, -1).transpose(0, 1), \
                y_pred.unsqueeze(0).expand(len_pred, -1) - y_pred.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

            exp_dist_pred_pred = torch.exp(
                -(x_dist_pred_pred ** 2 + y_dist_pred_pred ** 2) / (2 * smoothing_lambda ** 2))

            x_dist_pred_true, y_dist_pred_true = \
                x_pred.unsqueeze(0).expand(len_true, -1) - x_true.unsqueeze(0).expand(len_pred, -1).transpose(0, 1), \
                y_pred.unsqueeze(0).expand(len_true, -1) - y_true.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

            exp_dist_pred_true = torch.exp(
                -(x_dist_pred_true ** 2 + y_dist_pred_true ** 2) / (2 * smoothing_lambda ** 2))

            gradients = 0
            for channel_idx in range(location_pred.size()[2] - 2):
                p_pred = location_pred[batch_idx, :, 2 + channel_idx]
                p_pred_ex_tr = p_pred.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

                p_true = location_true[batch_idx, :, 2 + channel_idx]
                p_true_ex_tr = p_true.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

                xx = np.pi * p_pred * \
                     (torch.sum(p_true_ex_tr * exp_dist_pred_true * x_dist_pred_true, axis=0)
                      - torch.sum(p_pred_ex_tr * exp_dist_pred_pred * x_dist_pred_pred, axis=0))

                yy = np.pi * p_pred * \
                     (torch.sum(p_true_ex_tr * exp_dist_pred_true * y_dist_pred_true, axis=0)
                      - torch.sum(p_pred_ex_tr * exp_dist_pred_pred * y_dist_pred_pred, axis=0))

                pp = np.pi * smoothing_lambda ** 2 * \
                     (torch.sum(p_pred_ex_tr * exp_dist_pred_pred, axis=0)
                      - torch.sum(p_true_ex_tr * exp_dist_pred_true, axis=0))

                pp_mask = (location_pred.size()[2] - 2) * [0]
                pp_mask[channel_idx] = 1

                pp_gradients = [x * pp.unsqueeze(1) for x in pp_mask]
                gradients += torch.cat([xx.unsqueeze(1), yy.unsqueeze(1)] + pp_gradients, axis=1)

            FullGradient.append(gradients.unsqueeze(0))

        return torch.cat(FullGradient, axis=0) * grad, location_true * 0


# Sounting Loss
def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor 保证t在min-max之间
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = torch.as_tensor(t)
    t_min = torch.tensor(t_min)
    t_max = torch.tensor(t_max)
    result_out = (t >= t_min) * t + (t < t_min) * t_min
    result_out = (result_out <= t_max) * result_out + (result_out > t_max) * t_max
    return result_out


def corner_to_map(batch_y):
    x_offset = torch.zeros([batch_y.shape[0], 100, 100], dtype=torch.float)
    y_offset = torch.zeros([batch_y.shape[0], 100, 100], dtype=torch.float)
    score = torch.zeros([batch_y.shape[0], 100, 100], dtype=torch.float)
    for i in range(batch_y.shape[0]):
        for j in range(batch_y.shape[1]):
            row = int(batch_y[i, j, 1] // 4)
            col = int(batch_y[i, j, 0] // 4)
            # xx = (batch_y[i, j, 0] % 4.0) / 4.0
            # yy = (batch_y[i, j, 1] % 4.0) / 4.0
            # xx = torch.round(xx * 10000)/10000  # 保留4位小数
            # yy = torch.round(yy * 10000)/10000
            # x_offset[i, row, col] = xx
            # y_offset[i, row, col] = yy
            score[i, row, col] = 1.0
            # print("yy:{},row:{},float:{}--xx:{},col:{},float:{}".format(corner[i, 0, 1], row, round(yy, 3),
            #                                                             corner[i, 0, 0], col, round(xx, 3)))
    # print("OK1")
    return score  #, x_offset, y_offset


def ScoreLoss(p_pre, targets, threshold=0.05):
    # p_true, x_true, y_true = corner_to_map(targets)
    p_true = corner_to_map(targets)
    p_true = np.transpose(p_true, (0, 2, 1))
    p_true = p_true.reshape(targets.shape[0], -1)

    # pre_or = p_pre.permute(0, 2, 3, 1)
    # p_pred = pre_or[:, :, :, 2]
    # p_pred = p_pred.reshape(targets.shape[0], -1)
    p_pred = p_pre[:, :, 2]
    # 滤波
    # p_min = torch.tensor(threshold)
    # p_pred = (p_pred >= p_min) * p_pred + torch.zeros_like(p_pred)

    a = np.ones((targets.shape[0], 10000)) * 0.05
    p_true = torch.as_tensor(p_true, device=device)
    a = torch.tensor(a, device=device)
    score_loss = -torch.sum(torch.where(torch.gt(p_true, a),
                                        p_true * torch.log(clip_by_tensor(p_pred, 1e-6, 1.0)) / torch.sum(p_true) * 1,
                                        (torch.log(1 - clip_by_tensor(p_pred, 0, 0.999999)) / (
                                                targets.shape[0] * 10000 - torch.sum(p_true)))))

    # x_pred, y_pred = pre_or[:, :, :, 0], pre_or[:, :, :, 1]
    # # x_pred, y_pred = x_pred.reshape(targets.shape[0], -1), y_pred.reshape(targets.shape[0], -1)
    # # x_true, y_true = x_true.reshape(targets.shape[0], -1), y_true.reshape(targets.shape[0], -1)
    # x_true, y_true = torch.as_tensor(x_true, device=device), torch.as_tensor(y_true, device=device)
    # kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    # x_pre_soft = torch.softmax(x_pred, dim=-1)
    # x_true_soft = torch.softmax(x_true, dim=-1)
    # x_loss = kl_loss(x_pre_soft.log(), x_true_soft)
    # y_pre_soft = torch.softmax(y_pred, dim=-1)
    # y_true_soft = torch.softmax(y_true, dim=-1)
    # y_loss = kl_loss(y_pre_soft.log(), y_true_soft)
    #
    # offset_loss = x_loss+y_loss
    return score_loss  #, offset_loss


if __name__ == '__main__':

    y_true = np.load("F:\\chessboard\\origin\\synthesis\\0_corners.npy").astype(np.float32)
    y_true = np.squeeze(y_true, 1)
    y = torch.tensor(y_true)
    y = torch.unsqueeze(y, 0)
    xx = np.load("C:\\Users\\Huburt\\Desktop\\xx1.npy").astype(np.float32)
    x = torch.zeros(1,10000,3)
    x_n = torch.randn(10000)*0.4
    x[:,:,2] = torch.tensor(xx)*0.5 + x_n
    x=x.to(device)
    y=y.to(device)
    loss_count = ScoreLoss(x, y, threshold=0.5)
    print(loss_count)


