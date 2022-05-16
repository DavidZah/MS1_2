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
    frequencies = np.fft.fftfreq(len(spectrum), 1 / freq)
    i = frequencies > 0
    dom_freq = frequencies[np.argmax(abs(spectrum))]
    plt.plot(frequencies[i],2*power_spectrum[i])
    plt.xlabel("freq f [Hz]")
    plt.ylabel("|X(f)|")
    print(f'dominantni frkvence: {dom_freq}Hz')

def widows_fcn(singal):
    # okenkova funkce
    plt.figure(figsize=(15, 5))
    # mensi okenko
    plt.subplot(1, 2, 1)
    plt.title("Spektrogram s okénkem $2^8$")
    plt.xlabel("čas [s]")
    plt.ylabel("frekvence [Hz]")
    spectrogram1 = plt.specgram(signal, Fs=fs, NFFT=2 ** 8, mode="default", window=plt.mlab.window_hanning)
    # vetsi okenko
    plt.subplot(1, 2, 2)
    plt.title("Spektrogram s okénkem $2^{12}$")
    plt.xlabel("čas [s]")
    plt.ylabel("frekvence [Hz]")
    spectrogram2 = plt.specgram(signal, Fs=fs, NFFT=2 ** 12, mode="default", window=plt.mlab.window_hanning)
    plt.show()

def load_file():
    x = scipy.io.loadmat(file_name)
    signal = x["signal"]
    t = np.linspace(0, len(signal) - 1, len(signal)) * Ts
    plt.plot(t, signal)
    plt.show()
    return signal

def find_abnormal(signal):
    # nalezeni caso-frekvencnich udalosti
    spectrogram = spectrogram1
    # suma jednotlivych sloupcu
    column_sum = np.sum(spectrogram[0], axis=0)
    # vykresleni sum v jednotlivych casovych okamzicich
    plt.figure()
    plt.scatter(np.arange(0, (len(time) / 2 ** 12), (len(time) / 2 ** 12) / len(column_sum)) * (T * 2 ** 12),
                column_sum)
    # vypsani maxim
    print(
        f"casy ve kterych byla nalezena maxima: {np.argpartition(column_sum, -4)[-20:] * ((len(time) / 2 ** 12) / len(column_sum)) * (T * 2 ** 12)}")

if __name__ == '__main__':

    signal = load_file()
    signal = signal[:, 0]
    get_signal_info(signal)
    widows_fcn(signal)