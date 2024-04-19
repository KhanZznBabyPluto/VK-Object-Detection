import cv2

class ShapeDetector():

    def __init__(self):
        pass

    def detect(self, contour):
        """
        :param contour: contour that we got after find_contours() function
        :return: type of figure
        """

        shape = "unknown"
        curve = cv2.arcLength(contour, True)
        epsilon = 0.02
        curve_approx = curve * epsilon
        approx = cv2.approxPolyDP(contour, curve_approx, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            if ar >= 0.95 and ar <= 1.05:
                shape = "square"
            else:
                shape = "rectangle or parallelogram"
        else:
            shape = "circle"

        return shape
