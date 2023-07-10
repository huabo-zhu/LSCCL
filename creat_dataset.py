import glob
import os
import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import sklearn.datasets

if not os.path.exists("data/trian"):
    os.makedirs("data/trian")
if not os.path.exists("data/val"):
    os.makedirs("data/val")

# blur 0-1合适 超过1就很模糊
def addBlur(checkerboard, blur_level=None):
    if blur_level is None:
        blur_level = np.random.uniform(0, 1)
    # print(blur_level)
    warped_tmp = checkerboard + 5 * blur_level * np.random.randn(checkerboard.shape[0], checkerboard.shape[1],
                                                                 checkerboard.shape[2])  # noise
    warped = np.minimum(np.maximum(warped_tmp, 0), 255).astype(np.uint8)
    warped = cv2.GaussianBlur(warped, (7, 7), np.random.uniform(1, 2) * blur_level)  # blur

    return warped


# lighting
def addlighting(checkerboard):
    light = 0
    for kk in range(5):
        X, Y = np.meshgrid(np.arange(checkerboard.shape[1]), np.arange(checkerboard.shape[0]))
        pos = np.dstack((X, Y))
        mu = np.array([np.random.randint(checkerboard.shape[1]), np.random.randint(checkerboard.shape[0])])
        cov = checkerboard.shape[0] * checkerboard.shape[1] * \
              sklearn.datasets.make_spd_matrix(2, random_state=None) / np.random.randint(4, 10)
        rv = multivariate_normal(mu, cov)
        Z = rv.pdf(pos)

        light += (Z - np.min(Z)) * np.random.randint(50, 100) / (np.max(Z) - np.min(Z))
    light = np.minimum(light, 190)

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(light)
        plt.colorbar()
        fig.show()
    checkerboard_light = checkerboard.astype(np.float32) - (200 - light)[:, :, np.newaxis]
    checkerboard_light = np.round(np.maximum(checkerboard_light, 0)).astype(np.uint8)
    return checkerboard_light


# Image scaling
def scalingImage(checkerboard, real_corners, scaling=None):
    if scaling is None:
        scaling = np.random.uniform(0.5, 1.2, size=2)
    if not hasattr(scaling, "__len__"):
        scaling = (scaling, scaling)

    original = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], np.float32)[:, np.newaxis, :]
    projected = np.copy(original)
    projected[:, :, 0] *= scaling[0]
    projected[:, :, 1] *= scaling[1]

    M = cv2.getPerspectiveTransform(original.astype(np.float32), projected.astype(np.float32))
    warped = cv2.warpPerspective(checkerboard, M, (
        int(np.ceil(scaling[0] * checkerboard.shape[1])), int(np.ceil(scaling[1] * checkerboard.shape[0]))))
    w_points = cv2.perspectiveTransform(real_corners.reshape(-1, 1, 2), M)

    return warped, w_points


# Crop image
def crop_image(img, points):
    x_min = max(np.min(np.where(np.sum(img, axis=(1, 2)) != 0)) - 1, 0)
    x_max = np.max(np.where(np.sum(img, axis=(1, 2)) != 0)) + 1
    y_min = max(np.min(np.where(np.sum(img, axis=(0, 2)) != 0)) - 1, 0)
    y_max = np.max(np.where(np.sum(img, axis=(0, 2)) != 0)) + 1

    cropped = img[x_min:x_max, y_min:y_max]
    points -= np.array([y_min, x_min])[np.newaxis, np.newaxis, :]
    # cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cropped, points


# distortion
# if you want to use kd_tree, please install low version scikit-learn(my version 0.24.1)
def distortion(warped, points, dim, tree=False):
    def mapPoints(real_corners, mapx, mapy, tree=True):
        from sklearn.neighbors import KDTree
        undist_coords_f = np.concatenate([mapy.flatten()[:, np.newaxis], mapx.flatten()[:, np.newaxis]], axis=1)
        if tree:  # More efficient if many points to map
            # print('Initialize KDtree')
            tree = KDTree(undist_coords_f)
            # print('--Done')

        def calc_val(point_pos, shape_y):
            if tree:
                nearest_dist, nearest_ind = tree.query([point_pos], k=5)
                if nearest_dist[0][0] == 0:
                    return undist_coords_f[nearest_ind[0][0]]
            else:
                dist = np.linalg.norm(undist_coords_f - np.array(point_pos)[np.newaxis, :], axis=1)
                nearest_ind = np.argpartition(-dist, -5)[-5:]
                nearest_dist = dist[nearest_ind]

                idx_sort = np.argsort(nearest_dist)
                nearest_dist = nearest_dist[idx_sort]
                nearest_ind = nearest_ind[idx_sort]

            # starts inverse distance weighting
            w = np.array([1.0 / pow(d + 5e-10, 2) for d in nearest_dist])
            sw = np.sum(w)
            x_arr = np.floor(nearest_ind[0] / shape_y)
            y_arr = (nearest_ind[0] % shape_y)
            xx = np.sum(w * x_arr) / sw
            yy = np.sum(w * y_arr) / sw
            return (xx, yy)

        real_corners = real_corners.reshape(-1, 1, 2)
        # new_corners = np.zeros(real_corners.shape, np.float32)
        new_corners = np.zeros_like(real_corners, np.float32)
        for kk in range(real_corners.shape[0]):
            new_corners[kk, 0, [1, 0]] = calc_val([real_corners[kk, 0, 1], real_corners[kk, 0, 0]], mapy.shape[1])
        return new_corners

    # # 以前，前四个点是棋盘顶点
    # warped_points = np.zeros([points.shape[0]-4, points.shape[1]])
    # warped_points[:, :] = points[4:, :]
    # Distortion
    objp = np.zeros(((dim[1] - 1) * (dim[0] - 1), 3), np.float32)
    objp[:, :2] = np.mgrid[0:(dim[1] - 1), 0:(dim[0] - 1)].T.reshape(-1, 2).astype(np.float32)
    warped_points = points.copy()
    warped_points = warped_points.reshape(-1, 2).astype(np.float32)

    _, mtx, _, _, _ = cv2.calibrateCamera([objp], [warped_points], warped.shape[::-1], None, None)
    factor = 10
    mtx[0, 2] = factor * np.random.randint(85, warped.shape[1] - 85)
    mtx[1, 2] = factor * np.random.randint(85, warped.shape[0] - 85)
    mtx[0, 0] = factor * 1000
    mtx[1, 1] = factor * 1000 * warped.shape[0] / warped.shape[1]
    dis_scal = np.random.randint(5, 40)
    dist = np.array([dis_scal, 0, 0, 0, 0])

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx,
                                             (factor * warped.shape[1], factor * warped.shape[0]), 5)

    scaled_board, scaled_point = scalingImage(warped, warped_points, 10)

    dst = cv2.remap(scaled_board, mapx, mapy, cv2.INTER_LINEAR)
    dst_point = mapPoints(scaled_point, mapx, mapy, tree=True)
    out_img, out_pt = np.zeros_like(dst), np.zeros_like(dst_point)
    small_scale = 0.13
    for aa in range(4):
        scaled_board, scaled_point = scalingImage(dst, dst_point, small_scale)
        scaled_board = cv2.cvtColor(scaled_board, cv2.COLOR_GRAY2RGB)
        out_img, out_pt = crop_image(scaled_board, scaled_point)
        if out_img.shape[0] < 450 and out_img.shape[1] < 450:
            break
        small_scale -= float(aa + 1) * 0.005
    return out_img, out_pt  # .reshape(-1, 2)


# homography
def perspective(img, real_point, orientation):
    real_point = real_point.reshape(-1, 1, 2)
    # orientation:0 1 2 3 上下左右  小头朝向
    original = np.float32([[100, 100], [300, 100], [300, 300], [100, 300]]).reshape(-1, 1, 2)
    left_top = np.float32([100, 100])
    right_top = np.float32([300, 100])
    right_down = np.float32([300, 300])
    left_down = np.float32([100, 300])
    to_samll = np.random.randint(8, 40)
    to_big = np.random.randint(1, 8)
    xx = np.random.randint(8, 20)
    yy = np.random.randint(8, 20)
    xx1 = np.random.randint(8, 20)
    yy1 = np.random.randint(8, 20)
    if orientation == 0:  # 上头小
        left_top += np.float32([to_samll, xx])
        right_top += np.float32([-to_samll, yy])
        right_down += np.float32([to_big, -xx1])
        left_down += np.float32([-to_big, -yy1])
    if orientation == 1:  # 下头小
        left_top += np.float32([-to_big, xx])
        right_top += np.float32([to_big, yy])
        right_down += np.float32([-to_samll, -xx1])
        left_down += np.float32([to_samll, -yy1])
    if orientation == 2:  # 左头小
        left_top += np.float32([xx, to_samll])
        right_top += np.float32([-xx1, -to_big])
        right_down += np.float32([-yy1, +to_big])
        left_down += np.float32([yy, -to_samll])
    if orientation == 3:  # 右头小
        left_top += np.float32([xx, -to_big])
        right_top += np.float32([-xx1, to_samll])
        right_down += np.float32([-yy1, -to_samll])
        left_down += np.float32([yy, +to_big])
    projected = np.float32([left_top, right_top, right_down, left_down]).reshape(-1, 1, 2)
    M = cv2.getPerspectiveTransform(original, projected)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))
    warped_points = cv2.perspectiveTransform(real_point, M)
    return warped, warped_points


# rotation
def img_rotation_move(src, point):
    point = point.reshape(-1, 1, 2)
    theta = np.random.randint(-35, 35)
    move = np.float32([np.random.randint(-2, 2), np.random.randint(-2, 2)])
    height, width = src.shape[:2]  # 图片的高度和宽度
    # theta 顺时针旋转角度，单位为角度
    x0, y0 = width // 2, height // 2  # 以图像中心作为旋转中心
    MAR1 = cv2.getRotationMatrix2D((x0, y0), theta, 1.0)
    MAR1 = MAR1 + np.array([[0, 0, move[0]], [0, 0, move[1]]])
    imgR1 = cv2.warpAffine(src, MAR1, (width, height), borderValue=(255, 255, 255))  # 旋转变换，白色填充
    M = np.zeros([3, 3], dtype=np.float32)
    M[:2, :] = MAR1
    M[2, 2] = 1
    ratation_point = cv2.perspectiveTransform(point, M)
    return imgR1, ratation_point  # .reshape(-1, 2)


# draw chessboard
def draw_chessboard(col_num, row_num, img_height, img_width):
    margin = 10
    sz = np.array([img_height - margin * 2, img_width - margin * 2])
    num = max([col_num, row_num])
    square = min(sz[0], sz[1]) // num  # 单个方块最小尺寸
    all_square_size = np.array([col_num * square, row_num * square], dtype=int)
    start_col = int((img_width - all_square_size[0]) / 2 + 1)
    start_row = int((img_height - all_square_size[1]) / 2 + 1)
    chess = np.ones((img_height, img_width), dtype=np.uint8)

    xx, yy = np.meshgrid(square * np.arange(1, col_num) + start_col - 0.5,
                         square * np.arange(1, row_num) + start_row - 0.5)
    real_corners = np.concatenate([xx.flatten()[:, np.newaxis, np.newaxis],
                                   yy.flatten()[:, np.newaxis, np.newaxis]], axis=2).astype(np.float32)

    pixel = np.random.randint(170, 210)
    chess = chess * pixel
    for row in range(row_num):
        colr_grid = pixel
        start_y = start_row + row * square
        for col in range(col_num):
            start_x = start_col + col * square
            cv2.rectangle(chess, (start_x, start_y, square, square), colr_grid, -1)
            # next col
            colr_grid = 240 - colr_grid
        # next row
        pixel = 240 - pixel
    # chess = np.minimum(np.maximum(chess + np.random.randint(-10, 10, chess.shape), 0), 255).astype(np.uint8)
    # # 棋盘的四个顶点
    # all_point = np.zeros([real_corners.shape[0]+4, real_corners.shape[1], real_corners.shape[2]])
    # all_point[0][0] = np.array([(400 - img_width)/2, (400 - img_height)/2], dtype=np.float32)
    # all_point[1][0] = np.array([(400 - img_width)/2+img_width, (400 - img_height) / 2], dtype=np.float32)
    # all_point[2][0] = np.array([(400 - img_width)/2+img_width, (400 - img_height)/2+img_height], dtype=np.float32)
    # all_point[3][0] = np.array([(400 - img_width)/2, (400 - img_height)/2+img_height], dtype=np.float32)
    # all_point[4:,:,:] = real_corners
    # for idx in range(real_corners.shape[0]):
    #     all_point[idx+4][0] = real_corners[idx][0] + all_point[0][0]
    for idx in range(real_corners.shape[0]):
        real_corners[idx][0] += np.array([(400 - img_width) / 2, (400 - img_height) / 2], dtype=np.float32)

    return chess, real_corners, pixel


# blend texture
def blend_images(back_img, front_img, remove_edge_balck=True):
    if back_img is None or front_img is None:
        print("image is empty!")
        return 0
    org2 = front_img.copy()
    # front_img = np.array(front_img, dtype=np.uint8)
    rows, cols, channels = front_img.shape
    if remove_edge_balck:  # 去黑边
        allwhite = np.ones_like(front_img, dtype=np.uint8) * 255
        front_img = np.where(front_img == 0, allwhite, front_img)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        front_img = cv2.morphologyEx(front_img, cv2.MORPH_CLOSE, kernel1)

    roi = back_img[0:rows, 0:cols]
    # 原始图像转化为灰度值
    img2gray = cv2.cvtColor(front_img, cv2.COLOR_RGB2GRAY)
    # 将灰度值二值化，得到ROI区域掩模
    # cv2.threshold (源图片, 阈值=ret, 填充色, 阈值类型)
    ret, mask = cv2.threshold(img2gray, 241, 255, cv2.THRESH_BINARY)
    # ROI掩模区域反向掩模
    mask_inv = cv2.bitwise_not(mask)
    # 掩模显示背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    # 掩模显示前景
    img2_fg = cv2.bitwise_and(org2, org2, mask=mask_inv)
    # 前背景图像叠加
    dst = cv2.add(img1_bg, img2_fg)
    back_img[0:rows, 0:cols] = dst
    return back_img


if __name__ == "__main__":
    num_stat = 0
    num_count = 20000
    num_end = num_stat + num_count
    isSave = True
    isTrain = True
    if isTrain:
        background_img = './data/creat_dataset/train/*.jpg'
        save_path = "./data/trian"
    else:
        background_img = './data/creat_dataset/test/*.jpg'
        save_path = "./data/test"

    for idx in range(num_stat, num_end):
        w = np.random.randint(6, 11)
        h = np.random.randint(6, 10)
        if w == h:
            w += 1
        chess, point, _ = draw_chessboard(w, h, 230, 230)

        image = np.ones([400, 400], dtype=np.uint8)
        image = image * 255
        image[85:315, 85:315] = chess

        # 背景图像
        files_list = glob.glob(background_img)
        f_na = np.random.choice(files_list)
        background = cv2.imread(f_na)
        # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        # ave = np.average(background)
        # print(ave)

        warped = np.copy(image)
        warped_points = np.copy(point)
        img_save = np.copy(image)
        cor_save = np.copy(point)
        isBlend, isRemoveBlack = True, False
        # 透视变换
        if np.random.choice([True, False]):
            print("perspective")
            orient = np.random.randint(0, 4)
            warped, warped_points = perspective(warped, warped_points, orient)

        # 旋转平移
        if np.random.choice([True, False]):
            print("rotation_move")
            warped, warped_points = img_rotation_move(warped, warped_points)
        # 畸变
        try:
            if np.random.choice([True, True, False]):
                print("distortion")
                isRemoveBlack = True
                warped, warped_points = distortion(warped, warped_points, [w, h])
                if warped.shape[0] <= 400 and warped.shape[1] <= 400:
                    chess = np.ones([400, 400, 3], dtype=np.uint8) * 255
                    chess[:warped.shape[0], :warped.shape[1], :] = warped
                elif 400 < warped.shape[0] <= 500 and 400 < warped.shape[1] <= 500:
                    chess = np.ones([400, 400, 3], dtype=np.uint8) * 255
                    margin_x = int((warped.shape[1] - 400) / 2 - 1)
                    margin_y = int((warped.shape[0] - 400) / 2 - 1)
                    chess[:400, :400, :] = warped[margin_y:margin_y + 400, margin_x:margin_x + 400, :]
                    for aa in range(warped_points.shape[0]):
                        warped_points[aa, 0, :] -= [margin_x, margin_y]

                    # plt.figure()
                    # plt.imshow(chess)
                    # for i in range(warped_points.shape[0]):
                    #     points = warped_points.reshape(-1, 2)
                    #     center = [int(points[i, 0]), int(points[i, 1])]
                    #     plt.scatter(center[0], center[1], alpha=0.8, c='b')
                    #     plt.title("{}x{}".format(w, h))
                    # plt.savefig('./img/' + str(idx) + '_big.png')
                    # plt.close('all')
                else:
                    isBlend = False
                    print("out of border")
        except:
            print(idx, 'distortion failed')
        try:
            # 融合背景
            if isBlend:
                back_img = background.copy()
                if warped.ndim == 2:
                    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
                if back_img.shape[0] != 400 or back_img.shape[1] != 400:
                    back_img = cv2.resize(back_img, (400, 400))
                warped = blend_images(back_img, warped, isRemoveBlack)

            # 纹理噪声
            if np.random.choice([True, False, True]):
                noise_list = glob.glob('./data/creat_dataset/texture/*.jpg')
                n_na = np.random.choice(noise_list)
                noise = cv2.imread(n_na)
                noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB)
                ave = np.average(noise) / 255.0
                if noise.shape[0] != 400 or noise.shape[1] != 400:
                    noise = cv2.resize(noise, (400, 400))
                # if noise.shape != warped.shape:
                #     noise = cv2.resize(noise, (400, 400))
                warped = cv2.addWeighted(warped, 1, noise, ave * 0.5, 0)

        except:
            print(idx, 'blend or noise failed')

        #  add blur
        if np.random.choice([False, False, False]):
            warped = addBlur(warped)
        # add lighting
        if np.random.choice([False, False]):
            warped = addlighting(warped)

        if isSave:
            print(idx)
            cv2.imwrite('{}/{}.png'.format(save_path, idx), warped.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            np.save('{}/{}_corners.npy'.format(save_path, idx), warped_points.astype(np.float32))
            if not isTrain:
                dim = np.array([w - 1, h - 1])
                np.save('{}/{}_dim.npy'.format(save_path, idx), dim.astype(np.uint8))

