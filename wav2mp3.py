import os
import sys

wav_dir = 'audio'
mp3_dir = 'mp3'

try:
	os.mkdir(mp3_dir)
except:
	pass

# for wav_fn in os.listdir(wav_dir):
# 	wav_path = os.path.join(wav_dir, wav_fn)
# 	mp3_path = os.path.join(mp3_dir, wav_fn.split('.')[-2] + '.mp3')
# 	os.system('ffmpeg -i {} -vn -ar 44100 -ac 1 -b:a 192k {}'.format(wav_path, mp3_path))

# with open('mp3_file_list.txt', 'w') as writer:
# 	for mp3_fn in os.listdir(mp3_dir):
# 		writer.write('../mp3/' + mp3_fn + '\n')

with open('mp3_file_list.txt', 'w') as writer:
	for mp3_fn in os.listdir(wav_dir):
		writer.write('../audio/' + mp3_fn + '\n')
