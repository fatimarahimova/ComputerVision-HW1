import numpy as np
import cv2
import os
import moviepy.editor as mpy
from numpy.core.fromnumeric import nonzero
from numpy.lib.function_base import append

background=cv2.imread("HW1_material\Malibu.jpg")
#cv2.imshow('Background Image Window', background)
#cv2.waitKey(0)

background_height=background.shape[0]
background_width=background.shape[1]
ratio=360/background_height

background=cv2.resize(background, (int( background_width*ratio), 360))
print(background.shape)



image = cv2.imread('HW1_material\cat\cat_5.png')

foreground = np.logical_or(image[:,:,1]<180, image[:, :, 0] > 150)  # The pixels having cat image
nonzero_x, nonzero_y = np.nonzero(foreground)
nonzero_cat_values = image[nonzero_x, nonzero_y,:]
new_frame=background.copy()
new_frame[nonzero_x, nonzero_y, :]=nonzero_cat_values
new_frame[nonzero_x, -nonzero_y, :]=nonzero_cat_values
new_frame=new_frame[:,:,[2,1,0]]
cv2.imshow('cat',new_frame)
cv2.waitKey(0)
images_list = []
for i in range(0,180):
    image = cv2.imread('HW1_material\cat\cat_'+str(i)+'.png')
    foreground = np.logical_or(image[:,:,1]<180, image[:, :, 0] > 150)  # The pixels having cat image
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x, nonzero_y]
    new_frame=background.copy()
    new_frame[nonzero_x, nonzero_y, :]=nonzero_cat_values
    new_frame[nonzero_x, -nonzero_y, :]=nonzero_cat_values
    new_frame=new_frame[:,:,[2,1,0]]

    images_list.append(new_frame)

clip=mpy.ImageSequenceClip(images_list,fps=25)
audio=mpy.AudioFileClip('HW1_material\selfcontrol_part.wav').set_duration(clip.duration)
clip=clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video.mp4', codec= 'libx264')
