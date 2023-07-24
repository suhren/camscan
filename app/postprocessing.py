import cv2


def dummy(image: cv2.Mat) -> cv2.Mat:
    return image


def sharpen(image: cv2.Mat) -> cv2.Mat:
    blurred = cv2.GaussianBlur(src=image, ksize=(0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(src1=image, alpha=1.5, src2=blurred, beta=-0.5, gamma=0)
    return sharpened


def grayscale(image: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def black_and_white(image: cv2.Mat) -> cv2.Mat:
    gray = grayscale(image=image)
    sharpened = sharpen(image=gray)
    thresholded = cv2.adaptiveThreshold(
        src=sharpened,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=15,
    )
    return thresholded
