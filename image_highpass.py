import numpy as np
import cv2

def hifhpass_filter(src, a =0.5):

    src = np.fft.fft2(src)

    h, w = src.shape

    cy, cx = int(h/2), int(w/2)

    rh, rw = int(a*cy), int(a*cx)

    #第1象限と第３象限、第２象限と第４象限の入れ替え
    fsrc = np.fft.fftshift(src)

    #入力画像と同じサイズで値0の配列を生成
    fdst = fsrc.copy()

    #中心部分だけ0を代入（中心部分以外は元のまま）
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0

    #さっきずらした象限を元に戻す
    fdst = np.fft.fftshift(fdst)

    #高速逆フーリエ変換
    dst = np.fft.ifft2(fdst)

    #実部の値のみ取り出し、符号なし整数型に変換した返す。
    return np.uint8(dst.real)

def main():
    img = cv2.imread("C:/Users/kiyo/PycharmProjects/katagami_fft/images/arcKG00025.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gray, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    himg = hifhpass_filter(gray, 0.8)

    cv2.imwrite("output.png", himg)

if __name__ == "__main__":
    main()