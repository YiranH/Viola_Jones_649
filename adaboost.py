from haar import *

import numpy as np
import math

def learn_adaboost(pos_integral, neg_integral, classi_num, criterion,guess_matrix = None,skip_img = None, pre_classifiers = []):
    classifiers = []

    if skip_img != None:
        skip_img_len = len(skip_img)
    else:
        skip_img_len = 0

    pos_img_num = len(pos_integral)
    neg_img_num = len(neg_integral)
    img_integrals = pos_integral + neg_integral

    ground_truth_temp = [1] * pos_img_num + [0] * neg_img_num
    ground_truth = np.array(ground_truth_temp)

    pos_weights_temp = [1.0 / 2.0 * pos_img_num] * pos_img_num
    neg_weights_temp = [1.0 / 2.0 * neg_img_num] * neg_img_num
    weights = np.array(pos_weights_temp + neg_weights_temp)


    img_height, img_width = pos_integral[0].shape
    img_height -= 1
    img_width -= 1
    features, n_f = get_feature_list(img_height, img_width)

    if guess_matrix == None:
        guess = {}
        for i in range(len(img_integrals)):
            image = img_integrals[i]
            for j in range(len(features)):
                feature = features[j]
                guess[(i,j)] = feature.feature_classifier(image)
            if (i % 10 == 0):
                print('guessing: %d / %d' % (i, len(img_integrals)))
    else:
        guess = guess_matrix

    selected_idxes=set()

    if len(pre_classifiers) != 0:
        classifiers = pre_classifiers

    for t in range(len(pre_classifiers),classi_num):

        weights /= np.sum(weights)

        low_err = float('inf')
        selected_feature = None
        index = 0

        for i in range(len(features)):
            if i in selected_idxes:
                continue

            feature = features[i]

            err = 1e-8
            for j in range(len(img_integrals)):
                if skip_img!= None and j in skip_img:
                    continue

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


        training_accuracy = 0.0
        power = 0.0
        for j in range(len(img_integrals)):
            if skip_img != None and j in skip_img:
                continue

            if guess[(j,index)] == ground_truth[j]:
                power = 0.0
                training_accuracy += 1
            else:
                power = 1.0


            weights[j] = weights[j] * (low_err / (1.0 - low_err)) ** power

        selected_idxes.add(index)
        selected_feature.training_accuracy = training_accuracy / (len(img_integrals) - skip_img_len)
        selected_feature.alpha = math.log((1.0 - low_err) / low_err)
        classifiers.append(selected_feature)

        print('round: %d' % t)


    return classifiers



