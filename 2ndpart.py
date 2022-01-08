import numpy as np
import cv2
import os
import moviepy.editor as mpy
  
background=cv2.imread("HW1_material\Malibu.jpg")

background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360 / background_height

background = cv2.resize(background, (int(background_width*ratio), 360))

CatHistogramB = []
CatHistogramG = []
CatHistogramR = []
for i in range(0,180):
    img = cv2.imread(os.path.join('HW1_material\cat\cat_'+str(i)+'.png'))
    foreground = np.logical_or(img[:,:,1] < 180, img[:,:,0] > 150)
    histB = cv2.calcHist([img], [0], foreground.astype(np.uint8), [256], [0, 256]) #blue
    histG = cv2.calcHist([img], [1], foreground.astype(np.uint8), [256], [0, 256]) #green
    histR = cv2.calcHist([img], [2], foreground.astype(np.uint8), [256], [0, 256]) #red
    CatHistogramB.append(histB)
    CatHistogramG.append(histG)
    CatHistogramR.append(histR)
avg_cat_histogramB = np.mean(CatHistogramB, 0)
source_cdfB = avg_cat_histogramB.cumsum()
source_cdfB = 255 * (source_cdfB / source_cdfB.max())
source_cdfB = source_cdfB.astype("uint8")

avg_cat_histogramG = np.mean(CatHistogramG, 0)
source_cdfG = avg_cat_histogramG.cumsum()
source_cdfG = 255 * (source_cdfG / source_cdfG.max())
source_cdfG = source_cdfG.astype("uint8")

avg_cat_histogramR = np.mean(CatHistogramR, 0)
source_cdfR = avg_cat_histogramR.cumsum()
source_cdfR = 255 * (source_cdfR / source_cdfR.max())
source_cdfR = source_cdfR.astype("uint8")

target=cv2.imread("HW1_material/target_0.png")
foreground = np.logical_or(target[:,:,1] < 180, target[:,:,0] > 150)
targetB = cv2.calcHist([target], [0], None, [256], [0, 256]) #blue
targetG = cv2.calcHist([target], [1], None, [256], [0, 256]) #green
targetR = cv2.calcHist([target], [2], None, [256], [0, 256]) #red


target_cdfB = targetB.cumsum()
target_cdfB = 255 * (target_cdfB / target_cdfB.max())
target_cdfB = target_cdfB.astype("uint8")

target_cdfG = targetG.cumsum()
target_cdfG = 255 * (target_cdfG / target_cdfG.max())
target_cdfG = target_cdfG.astype("uint8")

target_cdfR = targetR.cumsum()
target_cdfR = 255 * (target_cdfR / target_cdfR.max())
target_cdfR = target_cdfR.astype("uint8")



def LUTfunc(cat,target):
    LUT=np.zeros((256,1), dtype="uint8")
    gj=0
    for gi in range(256):
        while(gj<256 and target[gj]<cat[gi]):
            gj=gj+1
        LUT[gi]=gj
    return LUT

LUT1=LUTfunc(source_cdfB, target_cdfB)
LUT2=LUTfunc(source_cdfG,target_cdfG)
LUT3=LUTfunc(source_cdfR,target_cdfR)

image_list = []
for i in range(0,180):
    image = cv2.imread(os.path.join('HW1_material\cat\cat_'+str(i)+'.png'))
    foreground = np.logical_or(image[:,:,1]<180, image[:, :, 0] > 150)  # The pixels having cat image
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x, nonzero_y]
    new_frame = background.copy()
    new_frame[nonzero_x,nonzero_y,:]  = nonzero_cat_values #To the previously obtained indices , the cat part is placed.
    
    for j in range(256):
        b_idx = np.where(nonzero_cat_values[..., 0] == j)
        g_idx = np.where(nonzero_cat_values[..., 1] == j)
        r_idx = np.where(nonzero_cat_values[..., 2] == j)


        nonzero_cat_values[b_idx, 0] = LUT1[j]
        nonzero_cat_values[g_idx, 1] = LUT2[j]
        nonzero_cat_values[r_idx, 2] = LUT3[j]
    
    new_frame[nonzero_x,background.shape[1] -1 -nonzero_y,:] = nonzero_cat_values #Mirror of cat image
    new_frame = new_frame[:,:,[2,1,0]]  #The frame here is currently in RGB order. However, the moviepy library defaultly uses BGR order. Thus, it may be good to reverse the channels.
    image_list.append(new_frame)
    new_frame_mirror = np.flip(new_frame,axis=1)

clip = mpy.ImageSequenceClip(image_list, fps = 25)
#print(clip.duration)
audio = mpy.AudioFileClip('HW1_material/selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip = audio)
clip.write_videofile('part2_video.mp4',codec ='libx264')

