from haar import *
from adaboost import *
from helpers import *

import numpy as np


def learn_cascade(pos_train, neg_train, max_t):

    F = np.ones(max_t + 10)

    cascaded_classifiers = []

    trains = pos_train + neg_train
    img_height, img_width = pos_train[0].shape
    img_height -= 1
    img_width -= 1
    features, n_f = get_feature_list(img_height, img_width)
    guess = {}
    for i in range(len(trains)):
        image = trains[i]
        for j in range(len(features)):
            feature = features[j]
            guess[(i, j)] = feature.feature_classifier(image)
        if (i % 10 == 0):
            print('guessing: %d / %d' % (i, len(trains)))

    skip_img = set()

    neg_train_len = len(neg_train)

    for t in range(1, max_t + 1):
        print('cascade round: %d' % t)

        F[t] = F[t - 1]

        feature_num  = 0

        classifier = []

        while F[t] >= F[t - 1]:
            feature_num += 1
            print('cascade feature: %d' % feature_num)

            classifier = learn_adaboost(pos_train, neg_train, feature_num, 'emp', guess_matrix=guess, skip_img = skip_img, pre_classifiers=classifier)

            accuracy, fp, fn, tp, tn, next_skip_img = find_acc(classifier, pos_train,neg_train,skip_img)

            F[t] = fp / neg_train_len

        if next_skip_img != None:
            skip_img = skip_img.union(next_skip_img)
            neg_train_len -= tn

        cascaded_classifiers.append(classifier)

    return cascaded_classifiers


