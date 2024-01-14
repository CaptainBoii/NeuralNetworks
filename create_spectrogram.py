from glob import glob

import subprocess

lengths = ['5s', '15s']
genres = ['Classical', 'Electronic', 'Hip-Hop', 'Metal', 'Pop', 'Rock']


def generate_spectrogram(input_file, output_file):
    subprocess.run(['ffmpeg', '-i', input_file, '-lavfi', 'showspectrumpic=s=250x250:legend=disabled', output_file])


if __name__ == '__main__':
    for length in lengths:
        for genre in genres:
            norm_audio_files = glob(length + '/' + genre + '/*.mp3')
            for input_audio_file in norm_audio_files:
                output_mel_spectrogram_file = 'Spectrograms/' + input_audio_file[:-3] + 'png'
                generate_spectrogram(input_audio_file, output_mel_spectrogram_file)
