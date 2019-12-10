from haar import *
from adaboost import *
from helpers import *
from cascade import *

import sys
from collections import defaultdict

import os
from PIL import Image

def part1():
    with open('part1.txt','wt') as file:
        sys.stdout = file
        _, f_num = get_feature_list(19,19)
        print('The total number of Haar Features is: %d.\n' % f_num['total'])
        for f_name in ["two_vertical", "two_horizontal", "three_vertical", "three_horizontal", "four"]:
            print('There are %d type %s features.' % (f_num[f_name], f_name))

def part2():
    pos_training_path = 'dataset/trainset/faces'
    neg_training_path = 'dataset/trainset/non-faces'
    pos_testing_path = 'dataset/testset/faces'
    neg_testing_path = 'dataset/testset/non-faces'


    train_poses = load_example(pos_training_path)
    train_negs = load_example(neg_training_path)
    test_poses = load_example(pos_testing_path)
    test_negs = load_example(neg_testing_path)
    train_poses_integral = [get_integral_image(image) for image in train_poses]
    train_negs_integral = [get_integral_image(image) for image in train_negs]
    test_poses_integral = [get_integral_image(image) for image in test_poses]
    test_negs_integral = [get_integral_image(image) for image in test_negs]

    classi_num = 10
    classifiers = learn_adaboost(train_poses_integral, train_negs_integral, classi_num, 'emp')




    with open('part2.txt','wt') as file:
        for round in [1,3,5,10]:
            top_acc = 0.0
            top_feature = None
            print('%d round detector' % round,file=file)
            for i in range(len(classifiers[:round])):
                feature = classifiers[i]
                print('Feature number %d:' % (i),file=file)
                print('Type: %s' % str(feature.type),file=file)
                print('Position: %s' % str(feature.t_left),file=file)
                print('Width: %d' % feature.width,file=file)
                print('Height: %d' % feature.height,file=file)
                print('Threshold: %f' % feature.threshold,file=file)
                print('Training accuracy: %f\n' % feature.training_accuracy,file=file)
                if (feature.training_accuracy > top_acc):
                    top_acc = feature.training_accuracy
                    top_feature = feature


            example =  test_poses[0]

            if top_feature.type == FEATURE[0]:
                t_left_np_1 = top_feature.t_left
                b_right_np_1 = (top_feature.t_left[0] + top_feature.width, int(top_feature.t_left[1] + top_feature.height / 2))
                t_left_np_2 = (top_feature.t_left[0], int(top_feature.t_left[1] + top_feature.height / 2))
                b_right_np_2 = top_feature.b_right
                
                example[t_left_np_1[1]: b_right_np_1[1], t_left_np_1[0]: b_right_np_1[0]] = 1.0
                example[t_left_np_2[1]: b_right_np_2[1], t_left_np_2[0]: b_right_np_2[0]] = 0.

            elif top_feature.type == FEATURE[1]:
                t_left_np_1 = top_feature.t_left
                b_right_np_1 = (int(top_feature.t_left[0] + top_feature.width / 2), top_feature.t_left[1] + top_feature.height)
                t_left_np_2 = (int(top_feature.t_left[0] + top_feature.width / 2), top_feature.t_left[1])
                b_right_np_2 = top_feature.b_right

                example[t_left_np_1[1]: b_right_np_1[1], t_left_np_1[0]: b_right_np_1[0]] = 1.
                example[t_left_np_2[1]: b_right_np_2[1], t_left_np_2[0]: b_right_np_2[0]] = 0.0

            elif top_feature.type == FEATURE[2]:
                t_left_np_1 = top_feature.t_left
                b_right_np_1 = (top_feature.b_right[0], int(top_feature.t_left[1] + top_feature.height / 3))
                t_left_np_2 = (top_feature.t_left[0], int(top_feature.t_left[1] + top_feature.height / 3))
                b_right_np_2 = (top_feature.b_right[0], int(top_feature.t_left[1] + 2 * top_feature.height / 3))
                t_left_np_3 = (top_feature.t_left[0], int(top_feature.t_left[1] + 2 * top_feature.height / 3))
                b_right_np_3 = top_feature.b_right

                example[t_left_np_1[1]: b_right_np_1[1], t_left_np_1[0]: b_right_np_1[0]] = 0.
                example[t_left_np_2[1]: b_right_np_2[1], t_left_np_2[0]: b_right_np_2[0]] = 1.0
                example[t_left_np_3[1]: b_right_np_3[1], t_left_np_3[0]: b_right_np_3[0]] = 0.

            elif top_feature.type == FEATURE[3]:
                t_left_np_1 = top_feature.t_left
                b_right_np_1 = (int(top_feature.t_left[0] + top_feature.width / 3), top_feature.t_left[1] + top_feature.height)
                t_left_np_2 = (int(top_feature.t_left[0] + top_feature.width / 3), top_feature.t_left[1])
                b_right_np_2 = (int(top_feature.t_left[0] + 2 * top_feature.width / 3), top_feature.t_left[1] + top_feature.height)
                t_left_np_3 = (int(top_feature.t_left[0] + 2 * top_feature.width / 3), top_feature.t_left[1])
                b_right_np_3 = top_feature.b_right

                example[t_left_np_1[1]: b_right_np_1[1], t_left_np_1[0]: b_right_np_1[0]] = 0.
                example[t_left_np_2[1]: b_right_np_2[1], t_left_np_2[0]: b_right_np_2[0]] = 1.0
                example[t_left_np_3[1]: b_right_np_3[1], t_left_np_3[0]: b_right_np_3[0]] = 0.

            elif top_feature.type == FEATURE[4]:
                t_left_np_1 = top_feature.t_left
                b_right_np_1 = (int(top_feature.t_left[0] + top_feature.width / 2), int(top_feature.t_left[1] + top_feature.height / 2))
                t_left_np_2 = (int(top_feature.t_left[0] + top_feature.width / 2), top_feature.t_left[1])
                b_right_np_2 = (top_feature.b_right[0], int(top_feature.t_left[1] + top_feature.height / 2))
                t_left_np_3 = (top_feature.t_left[0], int(top_feature.t_left[1] + top_feature.height / 2))
                b_right_np_3 = (int(top_feature.t_left[0] + top_feature.width / 2), top_feature.b_right[1])
                t_left_np_4 = (int(top_feature.t_left[0] + top_feature.width / 2), int(top_feature.t_left[1] + top_feature.height / 2))
                b_right_np_4 = top_feature.b_right

                example[t_left_np_1[1]: b_right_np_1[1], t_left_np_1[0]: b_right_np_1[0]] = 0.
                example[t_left_np_2[1]: b_right_np_2[1], t_left_np_2[0]: b_right_np_2[0]] = 1.0
                example[t_left_np_3[1]: b_right_np_3[1], t_left_np_3[0]: b_right_np_3[0]] = 0.
                example[t_left_np_4[1]: b_right_np_4[1], t_left_np_4[0]: b_right_np_4[0]] = 0.

            image = Image.fromarray(example * 255.).resize((100, 100))
            image = image.convert("L")
            image.save('image_of_round_' + str(round)+'.png')

            accuracy, fp, fn, tp, tn,_ = find_acc(classifiers[:round], test_poses_integral,test_negs_integral)



            print('Total accuracy: %f (%d/%d)' % (accuracy / len(test_poses_integral+test_negs_integral), accuracy, len(test_poses_integral+test_negs_integral)),file=file)
            print('False Positive: %f (%d/%d)' % (fp / len(test_negs_integral), fp, len(test_negs_integral)),file=file)
            print('False Negative: %f (%d/%d)\n\n' % (fn / len(test_poses_integral), fn, len(test_poses_integral)),file=file)



def part3():
    with open('part3.txt','wt') as file:
        for criterion in ['emp','fp','fn']:
            pos_training_path = 'dataset/trainset/faces'
            neg_training_path = 'dataset/trainset/non-faces'
            pos_testing_path = 'dataset/testset/faces'
            neg_testing_path = 'dataset/testset/non-faces'

            train_poses = load_example(pos_training_path)
            train_negs = load_example(neg_training_path)
            test_poses = load_example(pos_testing_path)
            test_negs = load_example(neg_testing_path)
            train_poses_integral = [get_integral_image(image) for image in train_poses]
            train_negs_integral = [get_integral_image(image) for image in train_negs]
            test_poses_integral = [get_integral_image(image) for image in test_poses]
            test_negs_integral = [get_integral_image(image) for image in test_negs]

            classi_num = 5
            classifiers = learn_adaboost(train_poses_integral, train_negs_integral, classi_num, criterion)



            accuracy, fp, fn, tp, tn,_ = find_acc(classifiers, test_poses_integral, test_negs_integral)


            print('Criterion: %s' % criterion,file=file)
            print('Total accuracy: %f (%d/%d)' % (accuracy / len(test_poses_integral + test_negs_integral), accuracy,
                                                 len(test_poses_integral + test_negs_integral)),file=file)
            print('False Positive: %f (%d/%d)' % (fp / len(test_negs_integral), fp, len(test_negs_integral)),file=file)
            print('False Negative: %f (%d/%d)\n' % (fn / len(test_poses_integral), fn, len(test_poses_integral)),file=file)


def part4():
    with open('part4.txt','wt') as file:
        pos_training_path = 'dataset/trainset/faces'
        neg_training_path = 'dataset/trainset/non-faces'
        pos_testing_path = 'dataset/testset/faces'
        neg_testing_path = 'dataset/testset/non-faces'

        train_poses = load_example(pos_training_path)
        train_negs = load_example(neg_training_path)
        test_poses = load_example(pos_testing_path)
        test_negs = load_example(neg_testing_path)
        train_poses_integral = [get_integral_image(image) for image in train_poses]
        train_negs_integral = [get_integral_image(image) for image in train_negs]
        test_poses_integral = [get_integral_image(image) for image in test_poses]
        test_negs_integral = [get_integral_image(image) for image in test_negs]

        classifiers = learn_cascade(train_poses_integral, train_negs_integral, 3)


        accuracy = 0.0
        tests = test_poses_integral + test_negs_integral
        drop_count = defaultdict(int)
        for i in range(len(tests)):
            image = tests[i]
            cas_guess = 1
            for j in range(len(classifiers)):
                classifier = classifiers[j]
                cop1 = 0.0
                cop2 = 0.0
                drop_count = 0.0
                for feature in classifier:
                    simple_guess = feature.feature_classifier(image)
                    cop1 += simple_guess * feature.alpha
                    cop2 += 0.5 * feature.alpha

                if cop1 >= cop2:
                    guess = 1
                else:
                    guess = 0

                if guess == 0:
                    drop_count[j] += 1
                    cas_guess = 0

            if i < len(test_poses_integral) and cas_guess == 1:
                accuracy += 1
            elif i < len(test_poses_integral) and cas_guess == 0:
                pass
            elif i >= len(test_poses_integral) and cas_guess == 0:
                accuracy += 1
            elif i >= len(test_poses_integral) and cas_guess == 1:
                pass

        print('Accuracy: %f' % (accuracy / len(tests)), file = file)
        for r in range(3):
            print('Round %d drops %d images' % (r, drop_count[r]))


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()

