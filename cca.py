import numpy as np
from scipy import signal
from sklearn.cross_decomposition import CCA


class classifier(object):
    """SSVEP classifier using Canonical Correlation Analysis (CCA).

    Generates reference signals for each stimulus frequency, then uses CCA
    to find the correlation between the EEG signal and each reference.
    The frequency with the highest correlation is selected as the output.

    Parameters
    ----------
    srate : int
        Sampling rate in Hz.
    display : str
        Display type. "mobile" uses a different frequency ordering.
    duration : float
        Duration (seconds) of the EEG segment used for classification.
    name : str
        Subject identifier.
    """

    def __init__(self, srate=500, display="mobile", duration=4, name="Liang"):
        self.srate = srate
        if display == "mobile":
            self.freqs = [7, 8, 9, 11, 7.5, 8.5]
        else:
            self.freqs = [7, 7.5, 8, 8.5, 9, 11]
        self.duration = duration
        self.name = name
        self.refsignal = self.init_refsignal().transpose((1, 2, 0))

    def bandpass_filter_signal(self, eeg, low=2, high=45, order=3):
        """Apply a Butterworth bandpass filter to the EEG signal."""
        nyq = 0.5 * self.srate
        low_c = low / nyq
        high_c = high / nyq
        b, a = signal.butter(order, [low_c, high_c], btype='band')
        y = signal.filtfilt(b, a, eeg)
        return y

    def init_refsignal(self):
        """Build reference signals for all stimulus frequencies."""
        freqRef = []
        for fr in range(len(self.freqs)):
            freqRef.append(self.getReferenceSignals(
                int(self.duration * self.srate), self.freqs[fr]))
        return np.stack(freqRef)

    def getReferenceSignals(self, length, target_freq, harmonics=4):
        """Generate sine/cosine reference signals for a target frequency.

        Parameters
        ----------
        length : int
            Number of data points.
        target_freq : float
            Stimulus frequency in Hz.
        harmonics : int
            Number of harmonics to include.

        Returns
        -------
        np.ndarray
            Reference signal matrix of shape (2*harmonics, length).
        """
        reference_signals = []
        t = np.arange(0, (length / self.srate), step=1.0 / self.srate)

        for i in range(harmonics):
            reference_signals.append(np.sin(np.pi * 2 * (i + 1) * target_freq * t))
            reference_signals.append(np.cos(np.pi * 2 * (i + 1) * target_freq * t))
        return np.array(reference_signals)

    def findCorr(self, signal_data, n_components=1):
        """Compute CCA correlation between the EEG and each reference signal.

        Returns
        -------
        np.ndarray
            Array of max correlation values, one per stimulus frequency.
        """
        freq = self.refsignal
        cca = CCA(n_components)
        corr = np.zeros(n_components)
        result = np.zeros(freq.shape[2])
        for freqIdx in range(freq.shape[2]):
            cca.fit(signal_data.T, np.squeeze(freq[:, :, freqIdx]).T)
            r_a, r_b = cca.transform(signal_data.T, np.squeeze(freq[:, :, freqIdx]).T)
            for indVal in range(n_components):
                corr[indVal] = np.corrcoef(r_a[:, indVal], r_b[:, indVal])[0, 1]
                result[freqIdx] = np.max(corr)
        return result

    def softmax(self, x):
        """Compute softmax probabilities."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_cca_cmd(self, eeg, n_components=1):
        """Run CCA on the last `duration` seconds of EEG and return (probs, command, rhos).

        Returns
        -------
        tuple
            (probabilities, predicted_class_index, raw_correlations)
        """
        timepoints = int(self.srate * self.duration)
        rhos = self.findCorr(eeg[:, -timepoints:], n_components=n_components)
        probs = self.softmax(rhos)
        cmd = np.argmax(probs)
        return probs, cmd, rhos

    def get_ssvep_command(self, eeg_signal):
        """Bandpass filter the EEG and return (predicted_class_index, raw_correlations)."""
        eeg = self.bandpass_filter_signal(eeg=eeg_signal)
        probs, cmd, rhos = self.get_cca_cmd(eeg)
        return cmd, rhos


if __name__ == '__main__':
    calib = classifier(name='tsai1', duration=3)
    print("Classifier initialized with frequencies:", calib.freqs)
