import os
from PIL import Image
import numpy as np

def load_example(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            example_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            example_arr /= example_arr.max()
            images.append(example_arr)
    return images

def find_acc(classifiers, test_poses_integral,test_negs_integral, skip_img = None):

    test_integral = test_poses_integral + test_negs_integral
    accuracy = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    tp = 0.0

    for i in range(len(test_integral)):
        if skip_img != None and i in skip_img:
            continue

        image = test_integral[i]
        cop1 = 0.0
        cop2 = 0.0
        for feature in classifiers:
            simple_guess = feature.feature_classifier(image)
            cop1 += simple_guess * feature.alpha
            cop2 += 0.5 * feature.alpha


        if cop1 >= cop2:
            guess = 1
        else:
            guess = 0

        if i < len(test_poses_integral) and guess == 1:
            tp += 1
            accuracy += 1
        elif i < len(test_poses_integral) and guess == 0:
            fn += 1
            if skip_img != None:
                skip_img.add(i)
        elif i >= len(test_poses_integral) and guess == 0:
            accuracy += 1
            tn += 1
            if skip_img != None:
                skip_img.add(i)
        elif i >= len(test_poses_integral) and guess == 1:
            fp += 1

    return accuracy, fp, fn, tp, tn, skip_img