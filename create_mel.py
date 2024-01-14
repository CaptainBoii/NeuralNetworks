import os
from glob import glob
import numpy as np
import librosa
from librosa import feature
import subprocess
import matplotlib.pyplot as plt

lengths = ['5s', '15s']
genres = ['Classical', 'Electronic', 'Hip-Hop', 'Metal', 'Pop', 'Rock']


def convert_audio(input_file, output_file):
    subprocess.run(['ffmpeg', '-i', input_file, '-ar', '22050', '-ac', '1', output_file])


def generate_mel_spectrogram(input_file, output_image, image_size=(250, 250)):

    # Convert audio to a temporary format
    temp_audio_file = 'temp_audio.wav'
    convert_audio(input_file, temp_audio_file)

    y, sr = librosa.load(temp_audio_file, sr=22050, mono=True)
    mel_spectrogram = feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Set the figure size
    plt.figure(figsize=(image_size[0] / 80, image_size[1] / 80), frameon=False)

    # Plot the pure mel spectrogram
    librosa.display.specshow(mel_spectrogram_db, x_axis=None, y_axis=None, sr=sr, fmax=8000)

    # Save the mel spectrogram as a PNG image
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, transparent=True)

    # Clean up temporary audio file
    os.remove(temp_audio_file)


if __name__ == '__main__':
    for length in lengths:
        for genre in genres:
            norm_audio_files = glob(length + '/' + genre + '/*.mp3')
            for input_audio_file in norm_audio_files:
                output_mel_spectrogram_file = 'MELs/' + input_audio_file[:-3] + 'png'
                generate_mel_spectrogram(input_audio_file, output_mel_spectrogram_file)
