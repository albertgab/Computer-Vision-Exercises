import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

directory = r"D:\OneDrive - Aberystwyth University\Projects\Vision\Images\\"


# os.chdir(directory)


def some_stats():
    i1 = cv2.imread(directory + "tree_dark_small.png", cv2.IMREAD_GRAYSCALE)
    print(np.max(i1), np.min(i1), np.average(i1))
    h1 = cv2.calcHist([i1], [0], None, [256], [0, 256])
    i2 = cv2.normalize(i1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    h2 = cv2.calcHist([i2], [0], None, [256], [0, 256])
    print(np.max(i2), np.min(i2), np.average(i2))
    cv2.imshow("s", i2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    fig, axes = plt.subplots(2, 2)  # creates 4 plotting axes
    axes[0, 0].imshow(i1, cmap="gray", vmin=0, vmax=255)  # shows img as is
    axes[0, 1].plot(h1)
    axes[1, 0].imshow(i2, cmap="gray", vmin=0, vmax=255)  # shows img_norm as is
    axes[1, 1].plot(h2)
    plt.show()


def threshold_fingerprint():
    i = cv2.imread(directory + "fingerprint.png", cv2.IMREAD_GRAYSCALE)
    i = cv2.bitwise_not(i)
    _, i_t = cv2.threshold(i, 140, 255, cv2.THRESH_BINARY)
    cv2.imshow("s", i_t)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def advert():
    i1 = cv2.imread(directory + "patricia.jpg")
    i2 = cv2.imread(directory + "logo.png", cv2.IMREAD_UNCHANGED)
    b, g, r, t = cv2.split(i2)
    i2 = cv2.merge((b, g, r))
    for i in range(i2.shape[0]):
        for j in range(i2.shape[1]):
            if t[i, j] == 255:
                i1[i + 10, j + 10] = i2[i, j]
    cv2.imshow("s", i1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def my_histogram(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist


def print_histogram():
    img = cv2.imread(directory + "fingerprint.png", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("s", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    my_hist = my_histogram(img)  # calculates histogram using your implementation
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(cv_hist)
    axes[0].set_title("OpenCV's")
    axes[1].plot(my_hist)
    axes[1].set_title("My histogram")
    fig.tight_layout()  # auto rearrange plots, to look nicer
    plt.show()
    fig.savefig("hist_compare.png", transparent=True)  # save plot with alpha


def filter_corridor():
    img = cv2.imread(directory + "corridor_noisy.png")
    print(img.shape)
    print(img.ndim)
    b, g, r = cv2.split(img)

    r = cv2.medianBlur(r, 3)
    img1 = cv2.merge((b, g, r))
    cv2.imshow("b", img)
    cv2.imshow("j", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(directory + "corridor_filtered.png", img1)


def obj_detection():
    img_org = cv2.imread(directory + "shelf.png")
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    temp = cv2.imread(directory + "template.png", cv2.IMREAD_GRAYSCALE)
    ncc = cv2.matchTemplate(img, temp, cv2.TM_CCORR_NORMED)
    cc = cv2.matchTemplate(img, temp, cv2.TM_CCORR)
    x, y, v = 0, 0, 0
    x1, y1, v1 = 0, 0, 0

    for i in range(ncc.shape[0]):
        for j in range(ncc.shape[1]):
            if v < ncc[i, j]:
                v = ncc[i, j]
                x = i
                y = j
    #to CHECK
    i, j = 0, 0
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            if v1 > ncc[i, j]:
                v1 = ncc[i, j]
                x1 = i
                y1 = j
    ncc = cv2.normalize(ncc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    c = int(x - temp.shape[1] / 2)
    cv2.rectangle(img_org, (y, x), (y + temp.shape[1], x + temp.shape[0]), (255, 0, 0), 8)
    cv2.imshow("s", img_org)
    cv2.imshow("j", ncc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    obj_detection()


if __name__ == '__main__':
    main()
