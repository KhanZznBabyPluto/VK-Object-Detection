import cv2
from not_a_model.contours import find_contours
from not_a_model.shape_detector import ShapeDetector

def square_counter(image):

    contours = find_contours(image)

    shape_detector = ShapeDetector()

    square_counter = 0

    if contours[-1][0][0][0] == 0 and contours[-1][0][0][1] == 0 and contours[-1][-1][0][1] == 0:
        contours = contours[0:-1]

    for contour in contours:
        if shape_detector.detect(contour) == "square":
            square_counter +=1

    print(square_counter)

    return

path = 'train_data/1.png'
square_counter(path)

