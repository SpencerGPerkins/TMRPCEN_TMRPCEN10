# Trainable Multi-rate PCEN (TMRPCEN)
_This is a companion repo for my graduate thesis: Trainable Multi-rate PCEN for Sound Event Detection from Field Recordings_
_This repo only includes the modules and scripts used during experimentation_

### Below is a brief description of the repo followed by the abstract from my thesis

## Bioacoustical (sic) Task
Scripts and modules used for experimentation on the [BirdVox-DCASE-20K](https://zenodo.org/records/1208080) Datset.
Modules include the frontend models (PCEN, MRPCEN, TMRPCEN, TMRPCEN10) and CNN built in Pytorch as well as some utilities.

## Urban Task
Scripts and modules used for experimentation on the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) Datset.
Modules include the frontend models (PCEN, MRPCEN, TMRPCEN, TMRPCEN10) and CNN built in Pytorch as well as some utilities.

## Frontend Models
+ PCEN : Computes Per-channel Energy Normalization as in [Wang et al](https://arxiv.org/abs/1607.05666) with learnable Automatic Gain Control (alpha parameter) and Dynamic Range Compresion (delta and r parameters).
+ MRPCEN : Computes Multi-rate Per-channel Energy Normalization similar to [Ick and McFee](https://arxiv.org/abs/2102.03468) with learnable Automatic Gain Control (alpha parameter) and Dynamic Range Compresion (delta and r parameters).
+ TMRPCEN : Includes learnability into the smoothing process within PCEN at multiple layers (rates) as in MRCPEN as in [Perkins et al](https://ieeexplore.ieee.org/document/10326212).
+ TMRPCEN : Less complex version of TMRPCEN with a single smoothing coefficient for each layer rather than different rates per frequency band at each layer.

# Abstract
> Sound is an important aspect of the world, providing an abundance of information, and accounting for a large portion of how the environments in which we move are perceived. Thus, research within the realm of sound and acoustic signals is of great importance. Recently, the use of methods such as deep learning and convolutional neural networks have aided in the pursuit of acoustical understanding, allowing for some alleviation of the difficulties within sonic research, namely, the time-consuming nature of working with acoustic signals and the resultant possibility of human error. However, within a deep learning approach, challenges with noise and simultaneous sound events still hamper the analysis of acoustic signals. This is especially true when the signal is obtained from a field recording such as in the bioacoustics and urban sound tasks that are explored in this study.
A typical approach to feature extraction from an audio sample consists of first obtaining a time-frequency representation of the signal. This is then transformed into the Mel scale followed by compression to reduce the dynamic range of the signal. Often, the final step consists of a logarithmic compression. Per-channel Energy Normalization (PCEN) is a recent method that replaces static logarithmic compression of the above third step. It has shown promise when dealing with background noise within an acoustic signal. PCEN has also been expanded by introducing PCEN computations at multiple layers, allowing for different levels of attenuation of background noise at different layers, known as Multi-rate PCEN (MRPCEN). Furthermore, this research introduces Trainable Multi-rate PCEN (TMRPCEN) which expands on these two methods by allowing for learnable functionality within all steps of the multi-rate PCEN process. These methods are utilized as trainable audio frontends and are compared through experiments on the BirdVox- DCASE-20K dataset and UrbanSound8K dataset, both containing acoustic signals obtained through field recordings.
The experimental results demonstrated a slightly higher performance of the TMRPCEN frontend on the former bioacoustics task, and comparable performance of all trainable frontends on the latter, urban sound task. In addition, all trainable frontends utilized in these experiments significantly outperformed the standard logarithmic compression approach. Ultimately, the results of the experiments within this study demonstrate the potential of learnable functionality within a single-rate and multi-rate approach to PCEN, allowing for better analysis of acoustic signals taken from field recordings, ultimately aiding in pursuits such as biological conservation and urban development.

