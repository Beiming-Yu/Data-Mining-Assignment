import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = ""
data_1 = pd.read_csv(data_path + "/train/100.csv")
data_1_tem = np.array(data_1.iloc[:, 1:3])
plt.plot(data_1_tem[:,0], data_1_tem[:,1], label="Wei")
data_2 = pd.read_csv(data_path + "/train/1000.csv")
data_2_tem = np.array(data_2.iloc[:, 1:3])
plt.plot(data_2_tem[:,0], data_2_tem[:,1], label="Ci")
data_3 = pd.read_csv((data_path + "/train/10000.csv")
data_3_tem = np.array(data_3.iloc[:, 1:3])
plt.plot(data_3_tem[:,0], data_3_tem[:,1], label="Tuo")
plt.legend()
plt.show()
