import os
import numpy as np
import scipy.io as scio


def load_GeoLife(name):
    with open(name, "r") as file:
        data = file.read().split('\n')[6:-1]
        mode = name.split("_")[1][:-4]
        latitude = np.array([item.split(',')[0] for item in data])
        longitude = np.array([item.split(',')[1] for item in data])
        location = np.concatenate((latitude, longitude))
        return location, mode


def load_TRAFFIC(data_path):
    data = scio.loadmat(data_path)
    labels = data['truth']
    data = data['tracks_traffic']
    mode = []
    location = []
    for i in range(len(labels)):
        if labels[i][1] == 1:
            mode.append(labels[i][0])
            tem = data[i][0].T
            location.append(tem)
    return np.array(location), np.array(mode)

# filenames = os.listdir("H:/下载/Geolife Trajectories 1.3/result/")
# for filename in filenames:
#     data, mode = load_GeoLife("H:/下载/Geolife Trajectories 1.3/result/" + filename)

# filename = "H:/下载/TRAFFIC/TRAFFIC.mat"
# data, mode = load_TRAFFIC(filename)
