import numpy as np
import cv2
import os
import moviepy.editor as moviepy

planes = np.zeros((9,472,4,3))

for i in range(1,10):
    with open("HW1_material\Plane_"+str(i)+".txt") as f:
        content = f.readlines()
        for line_id in range(len(content)):
            sel_line = content[line_id]
            sel_line = sel_line.replace(')\n', '').replace("(", '').split(")")

            for point_id in range(4):
                sel_point = sel_line[point_id].split(" ")

                planes[i-1,line_id,point_id,0] = float(sel_point[0])
                planes[i-1,line_id,point_id,1] = float(sel_point[1])
                planes[i-1,line_id,point_id,2] = float(sel_point[2])

images_list = []
album_img = cv2.imread("HW1_material/album.png")
album_img = cv2.resize(album_img, (572, 322))
album_img=cv2.cvtColor(album_img, cv2.COLOR_BGR2BGRA)
album_img[:,:,3]=255
album=np.float32([[0,0],[572, 0],[572,322],[0,322]])
catimg=cv2.imread("HW1_material/cat-headphones.png", cv2.IMREAD_UNCHANGED)
catimg=cv2.resize(catimg, (572,322))
catimg=cv2.cvtColor(catimg, cv2.COLOR_BGR2BGRA)

for i in range(472):
    blank_image = np.zeros((322,572,4), np.uint8)
    backImg = np.zeros((322,572,4), np.uint8)
    backImg[:,:,:]=255
    frontImg = catimg.copy()
    for j in range(9):
        index=0

        pts = planes[j,i,:,:].squeeze()[:,0:2].astype(np.int32)

        temp = np.copy(pts[3,:])
        pts[3, :] = pts[2,:]
        pts[2, :] = temp

        if pts[0][0] != pts[1][0]:
            pts = pts.reshape((-1, 1, 2))

            correspondence = np.zeros((8,8))
            for m in range(4):
                v1 = album[m][0]*pts[m][0][0]
                v2 = album[m][0]*pts[m][0][1]
                v3 = album[m][1]*pts[m][0][0]
                v4 = album[m][1]*pts[m][0][1]
                correspondence[index] = [pts[m][0][0], pts[m][0][1], 1, 0, 0, 0, -v1, -v2]
                correspondence[index+1] = [0, 0, 0, pts[m][0][0], pts[m][0][1], 1, -v3, -v4]
                index=index+2
            inverse=np.linalg.inv(correspondence)
            y = album.flatten()
            h = np.append(np.matmul(inverse,y), 1)
            h = h.reshape((3,3))
            result = cv2.warpPerspective(album_img,h,(572, 322), flags=cv2.WARP_INVERSE_MAP)
            foreground = np.logical_or(result[:,:,3] > 0, False)
            nonzero_x, nonzero_y = np.nonzero(foreground)
            nonzero_cat_values = result[nonzero_x, nonzero_y,:]
            if(pts[0][0][0]>pts[1][0][0]):
                backImg[nonzero_x,nonzero_y,:]= nonzero_cat_values
            else:
                frontImg[nonzero_x,nonzero_y,:]=nonzero_cat_values
    # cv2.imshow('test', frontImg)
    # cv2.waitKey(0)
    # cv2.imshow('test', backImg)
    # cv2.waitKey(0)
    
    foreground = np.logical_or(frontImg[:,:,3] == 0, False)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = backImg[nonzero_x, nonzero_y,:]
    frontImg[nonzero_x,nonzero_y,:]= nonzero_cat_values

    # cv2.imshow('test', frontImg)
    # cv2.waitKey(0)
    # break
    blank_image=frontImg[:,:,[2,1,0]]
    
    images_list.append(blank_image)


clip = moviepy.ImageSequenceClip(images_list, fps = 25)
clip.write_videofile("part3_vid.mp4", codec="libx264")