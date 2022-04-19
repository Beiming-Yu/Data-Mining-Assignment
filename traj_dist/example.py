import numpy as np
import traj_dist.distance as tdist
import pickle

# traj_list = pickle.load(open(r"D:\projects\traj-dist\data\benchmark_trajectories.pkl", "rb"))[:10]
# traj_list = np.array([[[1.0, 1.1], [1.2, 1.3], [1.0, 2.0]], [[1.2, 1.3], [1.4, 1.5]]])
# traj_A = traj_list[0]
# traj_B = traj_list[1]

traj_A = np.array([[1.0, 1.1], [1.2, 1.3], [1.0, 2.0]])
traj_B = np.array([[1.2, 1.3], [1.4, 1.5]])
traj_list = [traj_A, traj_B]
# Simple distance

dist = tdist.dtw(traj_A, traj_B)
print(dist)

# Pairwise distance

pdist = tdist.pdist(traj_list, metric="sspd")
print(pdist)

# Distance between two list of trajectories

cdist = tdist.cdist(traj_list, traj_list, metric="sspd")
print(cdist)
