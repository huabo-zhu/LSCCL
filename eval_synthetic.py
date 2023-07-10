import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils as utils
from model import *


def testDataset(file_path = "./data/val/*.png"):
    files_list = glob.glob(file_path)
    n_samples = len(files_list)
    data = np.zeros([n_samples, 400, 400, 3], np.uint8)
    y_loc = []
    dimension = []
    for kk in range(len(files_list[:n_samples])):
        data[kk] = cv2.imread(files_list[kk])
        y_loc.append(np.load(files_list[kk].replace('.png', '_corners.npy')).astype(np.float32))
        dimension.append(np.load(files_list[kk].replace('.png', '_dim.npy')).astype(np.float32))
    return data[:len(files_list[:n_samples])], y_loc, dimension, files_list

# load  dataset
print('>>>> Loading dataset')
X_data, y_location, y_dimension, file_list = testDataset()

Y_label = np.zeros([len(y_location), config['max_occurence'], dimPred])
for kk in range(len(y_location)):
    Y_label[kk, :y_location[kk].shape[0], :2] = y_location[kk][:, 0, :]
    Y_label[kk, :y_location[kk].shape[0], 2] = 1



model = myEfficientNet().to(device)
save_dict = torch.load('weight/20230609.ckpt',
                       map_location=lambda storage, location: storage)
model.load_state_dict(save_dict)
model.eval()

true_positive, false_negative, false_positive = 0, 0, 0
accuracy = []
xy_acc = np.zeros([1, 2], dtype=np.float32)
import time

t = time.time()
for idx_test in range(X_data.shape[0]):

    prediction = model(torch.tensor(X_data[idx_test:idx_test + 1], device=device).float())
    pp = prediction.cpu().data.numpy()

    pp = pp[0, pp[0, :, 2] > 0.2, :]
    pp = pp[:, [1, 0, 2]]

    # print(pp.shape)

    yy = Y_label[idx_test][Y_label[idx_test][:, 2] > 0.2, :]
    # yy -= 1
    if pp.shape[0] != 0:
        # quick cleaning: delete potential duplicates
        pp = pp[np.argsort(-pp[:, 2]), :]
        index_list = []
        for pp_index in range(pp.shape[0]):
            min_distance = 100
            for pp_compare in range(pp_index):
                min_distance = min(min_distance, np.linalg.norm(pp[pp_index, :2] - pp[pp_compare, :2]))
            # print(min_distance)
            if min_distance > 2:
                index_list.append(pp_index)
        pp = pp[np.array(index_list), :]

        if False:
            plt.figure(figsize=(15, 15))
            for jj in range(1):
                plt.subplot(1, 1, jj + 1)
                plt_img = cv2.cvtColor(X_data[idx_test + jj], cv2.COLOR_BGR2RGB)
                plt.imshow(plt_img)

                # Plot predictions
                for kk in range(config['n_channel']):
                    for ii in range(pp.shape[0]):
                        plt.scatter(pp[ii, 0], pp[ii, 1], alpha=pp[ii, 2 + kk], s=500,
                                    c='r')  # edgecolors="r", facecolors='r')

                plt.axis('equal')
                plt.gca().axis('off')
                plt.tight_layout()
            # plt.show()
            plt.savefig('output/plt/' + str(idx_test) + '.png')
            plt.close('all')

        distance = np.zeros([pp.shape[0], yy.shape[0]])
        for kk in range(pp.shape[0]):
            for jj in range(yy.shape[0]):
                distance[kk, jj] = np.linalg.norm(pp[kk, :2] - yy[jj, :2])

        mini = np.min(distance, axis=0)
        mini = mini[mini < 3]  # non-detection

        accuracy += mini.tolist()
        true_positive += np.sum(np.min(distance, axis=0) <= 3)
        false_negative += np.sum(np.min(distance, axis=0) > 3)

        # # yy来找pp中距离最近的数据，distance中0轴最小的为pp的索引
        # indexpp = distance.argmin(axis=0)
        # for aa, indx in enumerate(indexpp):
        #     temp = np.array([yy[aa, 0]-pp[indx, 0], yy[aa, 1]-pp[indx, 1]]).reshape(1, 2)
        #     xy_acc = np.append(xy_acc, temp, axis=0)

        assert ((pp.shape[0] - np.sum(np.min(distance, axis=0) <= 3)) >= 0)
        false_positive += (pp.shape[0] - np.sum(np.min(distance, axis=0) <= 3))

    else:
        false_negative += yy.shape[0]


elapsed = time.time() - t

print("--------------------------------")
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print(np.round(np.mean(accuracy), 3), np.round(recall * 100, 3),
      np.round(precision * 100, 3), np.round(2 * precision * recall / (precision + recall) * 100, 3))
print("percentile 25% 50% 75%: ", np.round(np.percentile(accuracy, [25, 50, 75]),3))
print(X_data.shape[0] / elapsed, " images per seconds")

print("time :", elapsed)

# np.savetxt("./x.txt", xy_acc[:, 0])
# np.savetxt("./y.txt", xy_acc[:, 1])