import numpy as np
import cv2

def main():
    img = cv2.imread("C:/Users/kiyo/PycharmProjects/katagami_fft/images/arcKG00025.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #高速フーリエ変換（二次元）
    fimg = np.fft.fft2(gray)

    print(fimg)
    #0周波数成分を配列の左上から中心に移動
    fimg = np.fft.fftshift(fimg)

    print(fimg)

if __name__ == "__main__":
    main()