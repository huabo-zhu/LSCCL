import glob
import cv2
import matplotlib.pyplot as plt
from model import *
threshold = 0.4


def getCorners(file_name):
    img = cv2.imread(file_name, 0)
    # img = cv2.GaussianBlur(img, (7, 7), 1)
    if len(img.shape) > 2:
        input = img[np.newaxis]
    else:
        input = np.tile(img[np.newaxis, :, :, np.newaxis], [1, 1, 1, 3])
    padded_input = np.zeros(
        [input.shape[0], int(4 * np.ceil(input.shape[1] / 4)), int(4 * np.ceil(input.shape[2] / 4)),
         input.shape[3]])
    padded_input[:, :input.shape[1], :input.shape[2], :] = input
    prediction = model(torch.tensor(padded_input, device=device).float())
    pp = prediction.cpu().data.numpy()
    pp = pp[0, pp[0, :, 2] > threshold, :]
    pp = pp[:, [1, 0, 2]]
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
    return pp


model = myEfficientNet().to(device)
save_dict = torch.load('weight/20230609.ckpt',
                       map_location=lambda storage, location: storage)
model.load_state_dict(save_dict)
model.eval()
img_path = './data/test/8.png'
corners = getCorners(img_path)

if True:
    plt.figure(figsize=(8, 8))
    for jj in range(1):
        plt.subplot(1, 1, jj + 1)
        img = cv2.imread(img_path, 0)
        plt.imshow(np.tile(img[:, :, np.newaxis], [1, 1, 3]))

        # Plot predictions
        point = corners
        # point = original_board
        for kk in range(1):
            for ii in range(point.shape[0]):
                plt.scatter(point[ii, 0], point[ii, 1], c='r')
        plt.axis('equal')
        plt.gca().axis('off')
        plt.tight_layout()
    plt.show()

print("x_coordinate   y_coordinate   score")
print(corners)
