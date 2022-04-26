import matplotlib.pyplot as plt
import numpy as np
import scipy.io
# Press the green button in the gutter to run the script.
import scipy as scipy
file_name = "signal.mat"
fs = 80000
Ts = 1/fs

def get_signal_info(signal):

    power = np.dot(np.transpose(signal),signal)*Ts
    mean=np.sum(signal)/len(signal)
    output_power = 


if __name__ == '__main__':
    x =  scipy.io.loadmat(file_name)
    signal = x["signal"]
    t = np.linspace(0, len(signal) - 1,len(signal)) * Ts
    plt.plot(t,signal)
    #plt.show()

    get_signal_info(signal)
