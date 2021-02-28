import os
import sys

wav_dir = 'audio'

with open('wav_file_list.txt', 'w') as writer:
	for wav_fn in os.listdir(wav_dir):
		writer.write('../audio/' + wav_fn + '\n')
