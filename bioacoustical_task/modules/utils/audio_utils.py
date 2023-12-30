# -*- coding: utf-8 -*-
"""
Created on Tues 11:25 2023

@author: Spencer Perkins

Module : Originally a module of different audio utils, currently
just a fancy way to open an audio file
"""

import librosa

#%% Utilities
class AudioUtils():

    """ Preprocessing utilities """

    """
    open: load the audio
    """

    def open(audio):

        """
        audio: a .wav file
        returns: the signal and sample rate
        """

        sig, sr = librosa.load(audio, sr=44100, mono=True)

        return (sig, sr)
