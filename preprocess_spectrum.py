#! python3

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


class SpectralFeature():
    """
    To preprocess the audio clips and extract some spectral features.
    """

    def __init__(self, audio_fname):
        """
        Load an audio clip and seperate harmonics and percussives into waveforms.
        """
        # Load the audio file
        y, sr = librosa.load(audio_fname)
        # Seperate harmonic and percussive sounds
        # Harmonic sound: pitched sound
        # Percussive sound: noise-like sound
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # Track beats
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # Save some attributes for later processing
        self.audio_fname = audio_fname
        self.y, self.sr = y, sr
        self.tempo = tempo
        self.beat_frames = beat_frames
        self.y_harmonic = y_harmonic
        self.y_percussive = y_percussive

        # Feature list
        self.features = {}

    def get_onset(self):
        """
        Locate note onset events. See [http://librosa.github.io/librosa/generated/librosa.onset.onset_detect.html#librosa.onset.onset_detect] for more details.
        """
        y, sr = self.y, self.sr

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr)

        self.features['onset_frames'] = onset_frames
        self.features['onset_env'] = onset_env
        self.features['onset_times'] = onset_times
        return onset_frames

    def get_mel_spectrograms(self):
        """
        Get mel-scaled spectrogram. See [http://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram] for more details.
        """
        y, sr = self.y, self.sr
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        self.features['melspec'] = melspec
        return melspec

    def get_mfcc(self, hop_length=512):
        """
        Get mel-frequency cepstral coefficients (MFCCs).
        """
        # MFCC
        y, sr = self.y, self.sr
        beat_frames = self.beat_frames

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        beat_mfcc_delta = librosa.util.sync(
            np.vstack([mfcc, mfcc_delta]), beat_frames)

        self.features['mfcc'] = mfcc
        self.features['mfcc_delta'] = mfcc_delta
        self.features['beat_mfcc_delta'] = beat_mfcc_delta
        return (mfcc, mfcc_delta, beat_mfcc_delta)

    def get_chromagram(self):
        """
        Get constant-Q chromogram.
        """
        y_harmonic, sr, beat_frames = self.y_harmonic, self.sr, self.beat_frames

        # Chroma features
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        beat_chroma = librosa.util.sync(
            chromagram, beat_frames, aggregate=np.median)

        self.features['chromagram'] = chromagram
        self.features['beat_chroma'] = beat_chroma
        return (chromagram, beat_chroma)

    def get_beat_features(self):
        """
        Collect all beat features that have been generated so far. This method must be called after other feature getters.
        """
        if 'beat_chroma' not in self.features.keys:
            return None
        elif 'beat_mfcc_delta' not in self.features.keys:
            return None

        beat_features = np.vstack(
            [self.features['beat_chroma'], self.features['beat_mfcc_delta']])

        self.features['beat_features'] = beat_features
        return beat_features

    def export_features(self):
        """
        Export all features that have been generated so far and save them as npy files along with the input audio clip.
        """
        for k, v in self.features.items():
            fname = '{}.feature.{}.npy'.format(self.audio_fname, k)
            np.save(fname, v)

    def run_all(self):
        """
        A wrapper for running all feature getters and export the results at once.
        """
        # self.get_mel_spectrograms()
        # self.get_mfcc()
        # self.get_chromagram()
        # self.get_beat_features()
        self.get_onset()

        self.export_features()


def extract_spectral_features(audio_fname):
    """
    Extract spectral features from audio_fname. All intermediate feature arrays are saved as npy files along with the input audio file (e.g., input.wav.beats.features.npy).

    Args:
        audio_fname: A filename string.
    """
    spectral_feature = SpectralFeature(audio_fname)
    spectral_feature.run_all()


# Itereate all audio clips and extract features
info_tab = pd.read_csv('data.v1.csv')
for i, audio_fname in enumerate(info_tab['audio_path']):
    print("Extracting features from {}...({}/{})".format(
        audio_fname, (i + 1), len(info_tab['audio_path'])))
    extract_spectral_features(audio_fname)
