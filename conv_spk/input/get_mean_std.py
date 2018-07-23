import numpy as np

feature_mean = np.load('../data/feature_mean.npy')
feature_std = np.load('../data/feature_std.npy')

feature_test_mean = np.load('../data/feature_test_mean.npy')
feature_test_std = np.load('../data/feature_test_std.npy')

print(feature_mean.shape)