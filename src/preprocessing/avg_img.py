import Image
from scipy.misc import imread, imsave, imresize
import os
import shutil
import numpy as np
import subprocess

def get_average_images(x_fname,x_videos_fname,img_dest,file_dest):
    """x_fname : source *.npy file
    x_videos_fname : *.npy file with videos numbers for x_fname
    img_dest : folder to save average images
    file_dest : file to save average images data in *.npy format
    """
    X_train_original = np.load("../../data/" + x_fname)
    X_train_videos = np.load("../../data/" + x_videos_fname)
    counter = 1
    videos = []
    for index in range(len(X_train_videos)-1):
        if X_train_videos[index] != X_train_videos[index+1]:
            counter += 1
            videos.append(X_train_videos[index])
    videos.append(X_train_videos[-1])
    frame = 0
    avg_images = np.array(np.zeros(shape = (counter,240,320,3) , dtype='uint64'))
    for video in range(counter):
        frames_count = 0
        while X_train_videos[frame] == X_train_videos[frame+1]:
            frames_count += 1
            avg_images[video] += X_train_original[frame]
            if (frame + 2) < len(X_train_videos):
                frame += 1
            else:
                break
        if (frame+3) < len(X_train_videos):
            frame += 1
        avg_images[video] = avg_images[video] / (frames_count)

    objects = os.listdir('../../data/')
    if img_dest in objects:
        shutil.rmtree('../../data/' + img_dest)
    os.makedirs('../../data/' + img_dest)

    for index in range(counter):  
        imsave('../../data/' + img_dest + '/' + "%03d" % index +'.jpg' , avg_images[index].astype('uint8'))
    np.save("../../data/" + file_dest , avg_images.astype('uint8'))