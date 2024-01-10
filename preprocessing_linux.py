import pandas as pd
import os
from pydub import AudioSegment

dir_base = "/run/media/bart/Dysk Muz/"

genres = pd.read_csv("genres.csv", sep=";", names=["Genre", "Id", "Name", "Path", "Grade"], encoding='utf-8')
for index, row in genres.iterrows():
    print(row["Name"])
    dir_ = dir_base
    if not pd.isna(row["Grade"]):
        dir_ += "「「「Rated/" + str(int(row["Grade"])) + "s/"
    dir_ += row["Path"]
    for file_count, file in enumerate(os.listdir(dir_)):
        file_format = ""
        if file[-3:] == "mp3":
            file_format = "mp3"
        elif file[-3:] == "m4a":
            file_format = "m4a"
        elif file[-4:] == "flac":
            file_format = "flac"

        if file_format != "":
            audio = AudioSegment.from_file(dir_+"/"+file, format=file_format)  # Replace with your file and format

            # Get the length of the audio in milliseconds
            audio_length = len(audio)

            if audio_length > 45 * 1000:
                # Define start and end time for the segment you want to extract (in milliseconds)
                start_time = 30 * 1000  # Replace with your start time
                end_time_1 = 45 * 1000  # Replace with your end time
                end_time_2 = 35 * 1000

                # Extract the segment between start and end time
                extracted_segment_1 = audio[start_time:end_time_1]
                extracted_segment_2 = audio[start_time:end_time_2]

                # Export the extracted segment as a new file (in the desired format)
                extract_path_1 = "15s/"+row["Genre"]+"/"+str(index+445)+"_"+str(file_count)+".mp3"
                extract_path_2 = "5s/"+row["Genre"]+"/"+str(index+445)+"_"+str(file_count)+".mp3"

                extracted_segment_1.export(extract_path_1, format="mp3")  # Replace with desired file name and format
                extracted_segment_2.export(extract_path_2, format="mp3")
