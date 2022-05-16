import matplotlib.pyplot as plt
import numpy as np
import scipy.io
# Press the green button in the gutter to run the script.
import scipy as scipy
file_name = "signal.mat"
fs = 80000
Ts = 1/fs

def get_signal_info(signal):
    mean_val = signal.mean()
    energy = np.dot(np.transpose(signal),signal)*Ts
    power = (1 / (Ts * len(signal))) * energy
    efective_val = np.sqrt(power)
    print(f"Střední hodnota {mean_val}\n"
          f"Energie {energy}\n"
          f"Vykon {power}\n"
          f"Efektivní hodnota {efective_val}")

def freq_parameters(signal):
    spectrum = np.fft.fft(signal)
    power_spectrum = abs(spectrum)
    frequencies = np.fft.fftfreq(len(spectrum), Ts)
    i = frequencies > 0
    dom_freq = frequencies[np.argmax(abs(spectrum))]
    plt.plot(frequencies[i],2*power_spectrum[i]/len(signal))
    plt.xlabel("freq f [Hz]")
    plt.ylabel("|X(f)|")
    plt.show()
    print(f'Dominantni frkvence: {dom_freq}Hz')

def widows_fcn(signal):
    # okenkova funkce
    plt.figure(figsize=(15, 5))
    # mensi okenko
    plt.subplot(1, 2, 1)
    plt.title("Spektrogram s okénkem 256")
    plt.xlabel("čas [s]")
    plt.ylabel("frekvence [Hz]")
    plt.specgram(signal, Fs=fs, NFFT=256, mode="default", window=plt.mlab.window_hanning)
    # vetsi okenko
    plt.subplot(1, 2, 2)
    plt.title("Spektrogram s okénkem 4096")
    plt.xlabel("čas [s]")
    plt.ylabel("frekvence [Hz]")
    plt.specgram(signal, Fs=fs, NFFT=4096, mode="default", window=plt.mlab.window_hanning)
    plt.show()

def load_file():
    x = scipy.io.loadmat(file_name)
    signal = x["signal"]
    t = np.linspace(0, len(signal) - 1, len(signal)) * Ts
    plt.plot(t, signal)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    return signal

def find_abnormal(signal):
    time = np.array(range(0, len(signal))) * Ts
    spectrogram = plt.specgram(signal, Fs=fs, NFFT=2 ** 8, mode="default", window=plt.mlab.window_hanning)
    plt.clf()
    column_sum = np.sum(spectrogram[0], axis=0)
    plt.scatter(np.arange(0, (len(time) / 4096), (len(time) / 4096) / len(column_sum)) * (Ts * 4096),
                column_sum)
    plt.show()
    max = np.argpartition(column_sum, -4)[-20:] * ((len(time) / 4096) / len(column_sum)) * (Ts *4096)
    print(f"Časy abnormalit: {max}")

if __name__ == '__main__':

    signal = load_file()
    signal = signal[:, 0]
    #get_signal_info(signal)
    freq_parameters(signal)
    #widows_fcn(signal)
    #find_abnormal(signal)
    print("done")