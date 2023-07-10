

import random
import matplotlib.pyplot as plt
import glob
import cv2
import os
import torch.optim as optim
import utils as utils
import Loss as losses
from model import *
from setting import *

if not os.path.exists("output"):
    os.makedirs("output")
if not os.path.exists("weight"):
    os.makedirs("weight")


# Generate training dataset
def training_dataset(file_path="./data/train/*.png", n_samples=20000):
    print('>>>> Loading dataset')
    img_data = np.zeros([n_samples, 400, 400, 3], np.uint8)
    y_location = []
    img_na = glob.glob(file_path)
    np.random.shuffle(img_na)
    img_name = img_na[0:n_samples]
    for iid, num in enumerate(img_name):
        img_data[iid] = cv2.imread(num)
        y_location.append(np.load(num.replace(".png", "_corners.npy")).astype(np.float32))

    Y_label = np.zeros([len(y_location), config['max_occurence'], dimPred])
    for kk in range(len(y_location)):
        Y_label[kk, :y_location[kk].shape[0], :2] = y_location[kk][:, 0, :]  # 复制真实点，加一个概率维度
        Y_label[kk, :y_location[kk].shape[0], 2] = 1  # 概率全部补1
    print('>>>> Loading backgrounds')
    # # Load background image
    background_images = []
    background_list = glob.glob(background_path)
    for file in background_list[:200]:
        im = cv2.imread(file)
        background_images.append(im)
    ##### Load test dataset
    print('>>>> Loading test data')
    ## MESA
    test_images = []
    test_list = glob.glob(test_path)
    np.random.shuffle(test_list)
    for test_img in test_list:
        im = cv2.imread(test_img)
        test_images.append(im)
    test_images = np.stack(test_images, axis=0)

    return img_data, Y_label, background_images, test_images


def eval_dataset(file_path="./data/val/*.png", n_samples=500):
    print('>>>> Loading eval dataset')
    img_data = np.zeros([n_samples, 400, 400, 3], np.uint8)
    y_location = []
    img_na = glob.glob(file_path + "*.png")
    np.random.shuffle(img_na)
    img_name = img_na[0:n_samples]
    for iid, num in enumerate(img_name):
        temp = cv2.imread(num)
        img_data[iid] = temp
        y_location.append(np.load(num.replace(".png", "_corners.npy")).astype(np.float32))

    Y_label = np.zeros([len(y_location), config['max_occurence'], dimPred])
    for kk in range(len(y_location)):
        Y_label[kk, :y_location[kk].shape[0], :2] = y_location[kk][:, 0, :]  # 复制真实点，加一个概率维度
        Y_label[kk, :y_location[kk].shape[0], 2] = 1  # 概率全部补1
    return img_data, Y_label


X_data, Y_label, background_img, test_img = training_dataset(trian_path, 2000)
eval_imgs, eval_cors = eval_dataset(val_path, 100)

model = myEfficientNet().to(device)
if isPretrian:
    save_dict = torch.load('weight/20230609.ckpt',
                           map_location=lambda storage, location: storage)
    model.load_state_dict(save_dict)
model.train()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # Adam

iter = 0
lossiter = 25000
lossiter1 = 5000
try:
    while iter <= config['niter']:
        if iter % 10 == 0:
            print(iter, np.round(max(min(0.1 * 5000 / max(iter - 25000, 1), 0.1), 0.01), 3))
        batch_x = np.zeros([config['batch_size'] + config['background_size'], X_data.shape[1], X_data.shape[2], 3],
                           np.uint8)
        batch_y = np.zeros(
            [config['batch_size'] + config['background_size'], config['max_occurence'], 2 + config['n_channel']])

        idx_batch = np.random.randint(0, len(X_data), config['batch_size'])  # 生成了4个随机数 为数组
        batch_x[:config['batch_size']] = np.transpose(X_data[idx_batch, :], (0, 2, 1, 3))  # 图像转置 xy交换
        batch_y[:config['batch_size']] = Y_label[idx_batch]
        for kk in range(config['background_size']):
            idx = np.random.randint(len(background_img))
            xmin = np.random.randint(background_img[idx].shape[0] - 100)
            ymin = np.random.randint(background_img[idx].shape[1] - 100)
            tmp = utils.random_transform(
                background_img[idx][xmin:xmin + X_data.shape[1], ymin:ymin + X_data.shape[2]])
            batch_x[config['batch_size'] + kk, :tmp.shape[0], :tmp.shape[1]] = tmp
        prediction = model(torch.tensor(batch_x, device=device).float())  # 训练时图像转置，所以训练输出是x,y,p,test时为y,x,p

        loss_count = losses.ScoreLoss(prediction, torch.tensor(batch_y, device=device),
                                      threshold=max(min(0.1 * lossiter1 / max(iter - lossiter, 1), 0.1), 0.01))

        loss_centroid = 0.7 * losses.torch_loss_centroid.apply(prediction,
                                                               torch.tensor(batch_y, device=device).float()) + 1

        if iter > lossiter:
            loss = loss_centroid + min((iter - lossiter) / lossiter, 1) * loss_count
        else:
            loss = loss_centroid

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss
        loss_window['loss'][1:] = loss_window['loss'][:-1]
        loss_window['loss'][0] = loss.cpu().data.numpy()
        list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

        loss_window['centroid'][1:] = loss_window['centroid'][:-1]
        loss_window['centroid'][0] = loss_centroid.cpu().data.numpy()
        list_loss['centroid'].append(np.median(loss_window['centroid'][loss_window['centroid'] != 0]))

        loss_window['count'][1:] = loss_window['count'][:-1]
        loss_window['count'][0] = loss_count.cpu().data.numpy()
        list_loss['count'].append(np.median(loss_window['count'][loss_window['count'] != 0]))

        loss_window['max'][1:] = loss_window['max'][:-1]
        loss_window['max'][0] = np.max(prediction[:, :, 2:].cpu().data.numpy())
        list_loss['max'].append(np.median(loss_window['max'][loss_window['max'] != 0]))
        if iter % 10 == 0 and iter > 0:
            print("heatmap loss:{}".format(loss_centroid.item()))
            print("Score loss:{}".format(loss_count.item()))

        # Figures and stuff
        if iter % 500 == 0 and iter > 0:
            print("-----", iter)
            plt.figure(figsize=(15, 9))
            plt.subplot(1, 4, 1)
            plt.plot(np.log(list_loss['loss']), 'k')
            plt.subplot(1, 4, 2)
            plt.plot(np.log(list_loss['centroid']), 'k')
            plt.subplot(1, 4, 3)
            plt.plot(np.log(list_loss['count']), 'k')
            plt.subplot(1, 4, 4)
            plt.plot(list_loss['max'], 'k')
            plt.ylim([0, 1.05])
            plt.savefig('output/loss.png')
            plt.close('all')

            model.eval()
            round_choice = np.random.choice(range(0, len(eval_imgs)), 4, replace=False)
            eval_data = []
            eval_pt = []

            for ii in round_choice:
                eval_data.append(eval_imgs[ii])
                eval_pt.append(eval_cors[ii])
            eval_data = np.stack(eval_data, axis=0)
            pred = model(torch.tensor(eval_data, device=device).float()).cpu().data.numpy()
            plt.figure(figsize=(15, 15))
            for kk in range(4):
                plt.subplot(2, 2, kk + 1)
                plt.imshow(eval_data[kk])
                for jj in range(config['n_channel']):
                    for ii in range(pred.shape[1]):
                        if pred[kk, ii, 2 + jj] > 0.2:
                            plt.scatter(pred[kk, ii, 1], pred[kk, ii, 0], alpha=pred[kk, ii, 2 + jj], c='r')
                plt.axis('equal')
                plt.gca().axis('off')
                plt.tight_layout()
            plt.savefig('output/eval.png')
            plt.close('all')
            testdata = eval_imgs[0:10]
            testcorner = eval_cors[0:10]
            # 输出精度
            true_positive, false_negative, false_positive = 0, 0, 0
            accuracy = []
            for idx_test in range(testdata.shape[0]):
                prediction = model(torch.tensor(testdata[idx_test:idx_test + 1], device=device).float())
                pp = prediction.cpu().data.numpy()
                pp = pp[0, pp[0, :, 2] > 0.2, :]
                pp = pp[:, [1, 0, 2]]
                yy = testcorner[idx_test][testcorner[idx_test][:, 2] > 0.2, :]
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
                    distance = np.zeros([pp.shape[0], yy.shape[0]])
                    for kk in range(pp.shape[0]):
                        for jj in range(yy.shape[0]):
                            distance[kk, jj] = np.linalg.norm(pp[kk, :2] - yy[jj, :2])
                    mini = np.min(distance, axis=0)
                    mini = mini[mini < 3]  # non-detection
                    accuracy += mini.tolist()
                    true_positive += np.sum(np.min(distance, axis=0) <= 3)
                    false_negative += np.sum(np.min(distance, axis=0) > 3)
                    assert ((pp.shape[0] - np.sum(np.min(distance, axis=0) <= 3)) >= 0)
                    false_positive += (pp.shape[0] - np.sum(np.min(distance, axis=0) <= 3))
                else:
                    false_negative += yy.shape[0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            print(np.round(np.mean(accuracy), 3), np.round(recall * 100, 1), np.round(precision * 100, 1),
                  np.round(2 * precision * recall / (precision + recall) * 100, 1))

            outstr = "\niter:{},acc:{},recall:{},precision:{},F:{}".format(iter, np.round(np.mean(accuracy), 3),
                                                                           np.round(recall * 100, 1),
                                                                           np.round(precision * 100, 1),
                                                                           np.round(2 * precision * recall / (
                                                                                       precision + recall) * 100, 1))
            with open('./output/eval.txt', 'a') as f:
                f.write(outstr)
            print("************************ eval ***********************")
            print(outstr)
            model.train()

        if iter % 2000 == 0 and iter > 0:
            torch.save(model.state_dict(), "weight/" + str(iter) + ".ckpt")
            model.eval()
            val_data = []
            for ii in range(4):
                random_index = random.randrange(len(test_img))
                val_data.append(test_img[random_index])
            val_data = np.stack(val_data, axis=0)
            pp = model(torch.tensor(val_data, device=device).float()).cpu().data.numpy()
            plt.figure(figsize=(15, 15))
            for kk in range(4):
                plt.subplot(2, 2, kk + 1)
                plt.imshow(val_data[kk])

                for jj in range(config['n_channel']):
                    for ii in range(pp.shape[1]):
                        if pp[kk, ii, 2 + jj] > 0.2:
                            plt.scatter(pp[kk, ii, 1], pp[kk, ii, 0], alpha=pp[kk, ii, 2 + jj], c='r')

                plt.axis('equal')
                plt.gca().axis('off')
                plt.tight_layout()
            plt.savefig('output/test.png')
            plt.close('all')
            model.train()

        iter += 1

        # try:
except:
    torch.save(model.state_dict(), "weight/" + "INTERRUPTED.ckpt")
    print('Saved interrupt')

