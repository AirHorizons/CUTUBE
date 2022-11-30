# pip install moviepy
import moviepy.editor as mp
import cv2

path = "./videos/"
filename = '4'

cap = cv2.VideoCapture(path+filename+".mp4")
FPS = cap.get(cv2.CAP_PROP_FPS)

clip = mp.VideoFileClip(path+filename+".mp4")
clip.audio.write_audiofile(path+filename+"_output.mp3")

my_clip = mp.VideoFileClip(path+filename+"_2"+".mp4")
audio_background = mp.AudioFileClip(path+filename+"_output.mp3")
final_clip = my_clip.set_audio(audio_background)
final_clip.write_videofile(path+filename+"_output.mp4",fps=FPS)
