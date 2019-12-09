from haar import *

import numpy as np
import math

# class adaboost(object):
#
#     def __init__(self, classifiers):
#         self.classifiers = classifiers
#
#     def adaboost_classifier(self,image):
#         for

def learn_adaboost(pos_integral, neg_integral, classi_num, criterion):
    classifiers = []

    pos_img_num = len(pos_integral)
    neg_img_num = len(neg_integral)
    img_integrals = pos_integral + neg_integral
    # num_imgs = len(img_integrals)

    ground_truth_temp = [1] * pos_img_num + [-1] * neg_img_num
    ground_truth = np.array(ground_truth_temp)

    pos_weights_temp = [1.0 / 2.0 * pos_img_num] * pos_img_num
    neg_weights_temp = [1.0 / 2.0 * neg_img_num] * neg_img_num
    weights = np.array(pos_weights_temp + neg_weights_temp)


    img_height, img_width = pos_integral[0].shape
    img_height -= 1
    img_width -= 1
    features, n_f = get_feature_list(img_height, img_width)
    # f_num = len(features)
    # feature_indexes = list(range(f_num))

    guess = {}
    for i in range(len(img_integrals)):
        image = img_integrals[i]
        for j in range(len(features)):
            feature = features[j]
            guess[(i,j)] = feature.feature_classifier(image)
        if (i % 10 == 0):
            print('guessing: %d / %d' % (i, len(img_integrals)))


    for t in range(classi_num):

        weights /= np.sum(weights)

        # classification_errors = np.zeros(f_num)
        low_err = float('inf')
        selected_feature = None
        index = 0

        for i in range(len(features)):
            feature = features[i]
        # for f in range(f_num):
        #     f_idx = feature_indexes[f]
            # classifier error is the sum of image weights where the classifier
            # is right
            err = 0.0
            for j in range(len(img_integrals)):
                if criterion == 'emp':
                    delta = weights[j] * abs(guess[(j,i)] - ground_truth[j])
                    err += delta
                elif criterion == 'fp':
                    if guess[(j,i)] == 1 and ground_truth[j] == 0:
                        delta = weights[j]
                        err += delta
                elif criterion == 'fn':
                    if guess[(j,i)] == 0 and ground_truth[j] == 1:
                        delta = weights[j]
                        err += delta

            if err < low_err:
                low_err = err
                selected_feature = feature
                index = i

            # if (i % 100 == 0):
                # print('erroring %d / %d' % (i,len(features)))
        # # get best feature, i.e. with smallest error
        # min_error_idx = np.argmin(classification_errors)
        # best_error = classification_errors[min_error_idx]
        # best_feature_idx = feature_indexes[min_error_idx]
        #
        # # set feature weight
        # best_feature = features[best_feature_idx]
        # feature_weight = 0.5 * np.log((1 - low_err) / low_err)
        # best_feature.weight = feature_weight

        training_accuracy = 0.0
        power = 0.0
        for j in range(len(img_integrals)):
            if guess[(j,index)] == ground_truth[j]:
                power = 0.0
                training_accuracy += 1
            else:
                power = 1.0
            weights[j] = weights[j] * (low_err / (1.0 - low_err)) ** power

        selected_feature.training_accuracy = training_accuracy / len(img_integrals)
        selected_feature.alpha = math.log((1.0 - low_err) / low_err)
        classifiers.append(selected_feature)

        print('round: %d' % t)


    return classifiers



