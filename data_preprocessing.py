import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def data_processing(train_file):
    """
    :param train_file: path to train file you want to split
    :return: two numpy arrays for train_data and test_data
    """

    data = pd.read_csv(train_file)

    path = data['img_path']
    label = data['label']
    type = data['type']

    train_paths, test_paths, train_labels, test_labels, train_types, test_types = train_test_split(path, label, type, test_size=0.2, random_state=42)
    
    train_data = []
    test_data = []

    # Проход по путям к изображениям, меткам и типам и добавление их в соответствующие списки
    for path, label, image_type in zip(train_paths, train_labels, train_types):
        train_data.append((path, label, image_type))

    for path, label, image_type in zip(test_paths, test_labels, test_types):
        test_data.append((path, label, image_type))

    # Преобразование списков в numpy массивы
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    return train_data, test_data

def image_processing(data):

    array = []

    for path in data:
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (32, 32))  # Изменение размера до 32x32
        array.append(resized_image)

    image_array = np.array(array)

    return image_array