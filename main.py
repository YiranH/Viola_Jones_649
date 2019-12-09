from haar import *
from adaboost import *
import sys
import pickle

import os
from PIL import Image

def load_example(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            example_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            example_arr /= example_arr.max()
            images.append(example_arr)
    return images

def find_acc(classifiers, test_poses_integral,test_negs_integral):

    test_integral = test_poses_integral + test_negs_integral
    for i in range(len(test_integral)):
        image = test_integral[i]
        cop1 = 0.0
        cop2 = 0.0
        accuracy = 0.0
        fp = 0.0
        fn = 0.0
        for feature in classifiers:
            simple_guess = feature.feature_classifier(image)
            cop1 += simple_guess * feature.alpha
            cop2 += 0.5 * feature.alpha

        if cop1 >= cop2:
            guess = 1
        else:
            guess = 0

        if i < len(test_poses_integral) and guess == 1:
            accuracy += 1
        elif i < len(test_poses_integral) and guess == 0:
            fp += 1
        elif i > len(test_poses_integral) and guess == 0:
            accuracy += 1
        elif i > len(test_poses_integral) and guess == 1:
            fn += 1

    return accuracy, fp, fn




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

    f = open('classifier_10', 'wb')
    pickle.dump(classifiers, f)
    f.close()

    with open('part2.txt','wt') as file:
        for round in [1,3,5,10]:
            top_acc = 0.0
            top_feature = None
            print('%d round detector' % round,file=file)
            for i in range(len(classifiers[:round])):
                feature = classifiers[i]
                print('Feature number %d:' % (i),file=file)
                print('Type: %s' % feature.name[3:-1],file=file)
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

            example = Image.fromarray(example * 255.).resize((100, 100))
            example.save('image_of_round_' + str(round))

            accuracy, fp, fn = find_acc(classifiers[:round], test_poses_integral,test_negs_integral)

            print('Total accuracy: %f (%d/%d)', (accuracy / len(test_poses_integral+test_negs_integral), accuracy, len(test_poses_integral+test_negs_integral)),file=file)
            print('False Positive: %f (%d/%d)', (fp / len(test_negs_integral), fp, len(test_negs_integral)),file=file)
            print('False Negative: %f (%d/%d)\n\n', (fn / len(test_poses_integral), fn, len(test_poses_integral)),file=file)



def part3():
    # with open('part3.txt','wt') as file:
    #     sys.stdout = file
        for criterion in ['fp', 'fn']:
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

            f = open('classifier_' + str(criterion), 'wb')
            pickle.dump(classifiers, f)
            f.close()

            # accuracy, fp, fn = find_acc(classifiers, test_poses_integral, test_negs_integral)
            # print('Criterion: %s' % criterion)
            # print('Total accuracy: %f (%d/%d)', (accuracy / len(test_poses_integral + test_negs_integral), accuracy,
            #                                      len(test_poses_integral + test_negs_integral)))
            # print('False Positive: %f (%d/%d)', (fp / len(test_negs_integral), fp, len(test_negs_integral)))
            # print('False Negative: %f (%d/%d)\n', (fn / len(test_poses_integral), fn, len(test_poses_integral)))




if __name__ == "__main__":
    # part1()
    # part2()
    part3()

