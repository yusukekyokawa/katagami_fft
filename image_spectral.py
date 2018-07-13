import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

save_path = os.path.abspath("C:/Users/kiyo/PycharmProjects/katagami_fft/result/spectrum/")
file_name = ".png"

all_img_path = glob.glob("./images/*.jpg")
for i, path in enumerate(all_img_path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fimg = np.fft.fft2(gray)

    fimg = np.fft.fftshift(fimg)

    mag = np.log(np.abs(fimg))

    plt.subplot(121)
    plt.imshow(gray, cmap = 'gray')
    plt.subplot(122)
    plt.imshow(mag, cmap='gray')


    plt.savefig(save_path+str(i)+file_name)