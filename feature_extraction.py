import csv
from glob import glob
import librosa
from librosa import feature
import numpy as np

lengths = ['5s', '15s']
genres = ['Classical Music', 'Electronic', 'Hip-Hop', 'Metal', 'Pop', 'Rock']

header = [
    'genre',
    'chroma_stft',
    'chroma_cqt',
    'chroma_cens',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'spectral_contrast',
    'mfcc',
    'melspectrogram',
    'tonnetz',
    'tempo',
    'tempogram',
    'fourier_tempogram',
    'rms',
    'zero_crossing_rate',
    'spectral_flatness'
]

fn_list_i = [
    feature.chroma_stft,
    feature.chroma_cqt,
    feature.chroma_cens,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff,
    feature.spectral_contrast,
    feature.mfcc,
    feature.melspectrogram,
    feature.tonnetz,
    feature.tempo,
    feature.tempogram,
    feature.fourier_tempogram,
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate,
    feature.spectral_flatness,
]


def get_feature_vector(y, sr, category):
    feat_vect_i = [np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y=y)) for funct in fn_list_ii]
    f_v = [category] + feat_vect_i + feat_vect_ii
    return f_v


for length in lengths:
    norm_audios_feat = []
    norm_output = length + '_normals.csv'
    for genre in genres:
        norm_audio_files = glob(length + '/' + genre + '/*.mp3')
        for file in norm_audio_files:
            y, sr = librosa.load(file, sr=None)
            feature_vector = get_feature_vector(y, sr, genre)
            norm_audios_feat.append(feature_vector)

    with open(norm_output, '+w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(header)
        csv_writer.writerows(norm_audios_feat)
