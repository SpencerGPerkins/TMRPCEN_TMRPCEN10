# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:56:18 2023

@author: Spencer perkins

Module : Initial Feature Extraction methods (stft, mel-spec, log compression)
"""

import numpy as np
import librosa
import librosa.display

class featureExtractor:
    """Feature Extraction
    by
    Fourier Transformations/pooling/compression
    """

    def get_features(audio,
                     features : str
                     ):
        """
        Parameters
        ----------
        audio : numpy.array
            Audio file.
        features : str
            stft, melspec, logmelspec

        Returns
        -------
        A filterbank of the audio file
        """

        sig, sr = audio # Signal and sample rate

        if features == 'melspec' or 'logmelspec':
            mel_spec = librosa.feature.melspectrogram(y=sig,
                                                      sr=44100,
                                                      n_fft=1024,
                                                      hop_length=1024//2,
                                                      win_length=1024,
                                                      n_mels=128,
                                                      power=1,
                                                      fmin=0,
                                                      fmax=sr/2.0
                                                      )

            if features == 'logmelspec': # Log compression
                out =librosa.amplitude_to_db(mel_spec, ref=np.max)
                return out
            else:
                out = mel_spec
                return out

        elif features == 'stft': # Vanilla stft
            filterbank = librosa.stft(y=sig,
                                      n_fft=1024,
                                      hop_length=1024//2,
                                      win_length=1024
                                      )
            filterbank = np.abs(filterbank)
            return filterbank

        else:
            raise ValueError(
                'Feature Extraction method inputs are: stft, melspec, logmelspec'
                )
