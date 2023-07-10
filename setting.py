import torch
import numpy as np

isPretrian = True  # Whether to load the pre_trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {'learning_rate': 0.001,  # 0.001-->0.0001
          'batch_size': 8, 'background_size': 0, 'niter': 80000, 'n_channel': 1,
          'max_occurence': 110}
# Loss placeholders
loss_window = {'loss': np.zeros([100]), 'max': np.zeros([100]),
               'centroid': np.zeros([100]), 'count': np.zeros([100])}
list_loss = {'loss': [], 'max': [], 'centroid': [], 'count': []}
dimPred = 3
n_points = 1
smoothing_lambda = 1.25

# image path
trian_path = 'F:\\chessboard\\train\\train400\\real\\*.png'   # Training Data Folder
background_path = 'E:\\Code\\DL\\ACCVchessboarod\\Checkerboard\\background\\train\\*.jpg'  # Background Image Folder
test_path = 'E:\\Code\\DL\\ACCVchessboarod\\Checkerboard\\benchmarkData\\val\\*.jpg'  # Real Test Image Folder
val_path = 'F:\\chessboard\\train\\train400\\val\\*.png'  # Val Image Folder
