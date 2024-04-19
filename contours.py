import cv2

def find_contours(image_path):
    image = cv2.imread(image_path)

    shifted = cv2.pyrMeanShiftFiltering(image, 3, 90)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    cv2.imshow("Image", image)
    cv2.waitKey()

    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    cv2.namedWindow("Contours Detection", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Contours Detection", image)
    cv2.waitKey()

    return contours

# path = 'train_data/2.png'
# find_contours(path)