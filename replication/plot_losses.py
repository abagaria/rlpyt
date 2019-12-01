import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
if __name__=='__main__':
    envs = ["Ant-v2", "Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
    plot_name = ["Inverse Dynamics Loss", "Transition Dynamics Loss"]
    dirname = 'data/EAC/losses'
    os.mkdir(dirname + "/figs")
    for j,e in enumerate(envs):
        with open(dirname + "/" + e + "-losses.pkl", "rb") as f:
            t, inv, trans = pickle.load(f)
            for i, (loss, name) in enumerate(zip((inv, trans), plot_name)):
                mean = np.mean(loss, axis=0)
                std = np.std(loss, axis=0)
                plt.figure(j*2 + i)
                plt.plot(t, mean)
                plt.fill_between(t, mean-std, mean+std, alpha=0.3)
                plt.title(e + ": " + name)
                plt.xlabel('Environment Steps')
                plt.savefig(dirname + "/figs/" + e + "-" + name + ".pdf")
    plt.show()
