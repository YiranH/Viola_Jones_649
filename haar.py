import numpy as np

def get_integral_image(img_array):


    sum_by_row = np.zeros(img_array.shape)
    # we need an additional column and row
    row = img_array.shape[0]
    column = img_array.shape[1]
    integral_array = np.zeros((row + 1, column + 1))
    for x in range(column):
        for y in range(row):
            sum_by_row[y, x] = sum_by_row[y-1, x] + img_array[y, x]
            integral_array[y+1, x+1] = integral_array[y+1, x] + sum_by_row[y, x]
    return integral_array





FEATURE = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2))


class HaarFeature(object):
    """
    class for haar-like feature
    """

    def __init__(self, f_type, pos, width, height, threshold, parity):
        """
        initialize a new haar-like feature
        """
        self.type = f_type
        self.t_left = pos
        self.b_right = (pos[0] + width, pos[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.parity = parity
        self.alpha = 0.0
        self.training_accuracy = 0.0

    def get_sum(self, integral_array, t_left_np, b_right_np):
        """
        calculates the sum in the rectangle by feature type
        |----------------|
        |t_left   t_right|
        |                |
        |b_left   b_right|
        |----------------|
        :return The sum of all pixels in the given rectangle
        :rtype int
        """
        t_left = (t_left_np[1], t_left_np[0])
        b_right = (b_right_np[1], b_right_np[0])
        b_left = (b_right[0], t_left[1])
        t_right = (t_left[0], b_right[1])

        sum = integral_array[b_right] + integral_array[t_left] - integral_array[t_right] -\
              integral_array[b_left]

        return sum

    def get_score(self, integral_array):

        diff = 0
        if self.type == FEATURE[0]:
            t_left_np_1 = self.t_left
            b_right_np_1 = (self.t_left[0] + self.width, int(self.t_left[1] + self.height / 2))
            t_left_np_2 = (self.t_left[0], int(self.t_left[1] + self.height / 2))
            b_right_np_2 = self.b_right
            sum_1 = self.get_sum(integral_array, t_left_np_1, b_right_np_1)
            sum_2 = self.get_sum(integral_array, t_left_np_2, b_right_np_2)
            diff = sum_1 - sum_2

        elif self.type == FEATURE[1]:
            t_left_np_1 = self.t_left
            b_right_np_1 = (int(self.t_left[0] + self.width / 2), self.t_left[1] + self.height)
            t_left_np_2 = (int(self.t_left[0] + self.width / 2), self.t_left[1])
            b_right_np_2 = self.b_right
            sum_1 = self.get_sum(integral_array, t_left_np_1, b_right_np_1)
            sum_2 = self.get_sum(integral_array, t_left_np_2, b_right_np_2)
            diff = sum_1 - sum_2

        elif self.type == FEATURE[2]:
            t_left_np_1 = self.t_left
            b_right_np_1 = (self.b_right[0], int(self.t_left[1] + self.height / 3))
            t_left_np_2 = (self.t_left[0], int(self.t_left[1] + self.height / 3))
            b_right_np_2 = (self.b_right[0], int(self.t_left[1] + 2 * self.height / 3))
            t_left_np_3 = (self.t_left[0], int(self.t_left[1] + 2 * self.height / 3))
            b_right_np_3 = self.b_right
            sum_1 = self.get_sum(integral_array, t_left_np_1, b_right_np_1)
            sum_2 = self.get_sum(integral_array, t_left_np_2, b_right_np_2)
            sum_3 = self.get_sum(integral_array, t_left_np_3, b_right_np_3)
            diff = sum_1 - sum_2 + sum_3

        elif self.type == FEATURE[3]:
            t_left_np_1 = self.t_left
            b_right_np_1 = (int(self.t_left[0] + self.width / 3), self.t_left[1] + self.height)
            t_left_np_2 = (int(self.t_left[0] + self.width / 3), self.t_left[1])
            b_right_np_2 = (int(self.t_left[0] + 2 * self.width / 3), self.t_left[1] + self.height)
            t_left_np_3 = (int(self.t_left[0] + 2 * self.width / 3), self.t_left[1])
            b_right_np_3 = self.b_right
            sum_1 = self.get_sum(integral_array, t_left_np_1, b_right_np_1)
            sum_2 = self.get_sum(integral_array, t_left_np_2, b_right_np_2)
            sum_3 = self.get_sum(integral_array, t_left_np_3, b_right_np_3)
            diff = sum_1 - sum_2 + sum_3

        elif self.type == FEATURE[4]:
            t_left_np_1 = self.t_left
            b_right_np_1 = (int(self.t_left[0] + self.width / 2), int(self.t_left[1] + self.height / 2))
            t_left_np_2 = (int(self.t_left[0] + self.width / 2), self.t_left[1])
            b_right_np_2 = (self.b_right[0], int(self.t_left[1] + self.height / 2))
            t_left_np_3 = (self.t_left[0], int(self.t_left[1] + self.height / 2))
            b_right_np_3 = (int(self.t_left[0] + self.width / 2), self.b_right[1])
            t_left_np_4 = (int(self.t_left[0] + self.width / 2), int(self.t_left[1] + self.height / 2))
            b_right_np_4 = self.b_right
            # top left
            sum_1 = self.get_sum(integral_array, t_left_np_1, b_right_np_1)
            # top right
            sum_2 = self.get_sum(integral_array, t_left_np_2, b_right_np_2)
            # bottom left
            sum_3 = self.get_sum(integral_array, t_left_np_3, b_right_np_3)
            # bottom right
            sum_4 = self.get_sum(integral_array, t_left_np_4, b_right_np_4)
            diff = sum_1 - sum_2 - sum_3 + sum_4
        return diff

    def feature_classifier(self, img_array):
        diff = self.get_score(img_array)
        if self.parity * diff < self.parity * self.threshold:
            return 1
        else:
            return 0

def get_feature_list(img_width, img_height):
    '''
    generate all possible features for given image size and max feature size
    '''
    f_names = ["two_vertical", "two_horizontal", "three_vertical", "three_horizontal", "four"]
    f_list = []
    max_f_width = 8
    max_f_height = 8

    f_num = {}

    for i in range(5):
        f_cnt = 0  # number of features of certain type
        f_min_width = FEATURE[i][0]
        f_min_height = FEATURE[i][1]
        f_name = f_names[i]

        # feature size with unit size of (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)
        for f_width in range(f_min_width, max_f_width + 1, f_min_width):
            for f_height in range(f_min_height, max_f_height + 1, f_min_height):

                # different possible positions (indexed by top left corner)
                for x in range(img_width - f_width + 1):
                    for y in range(img_height - f_height + 1):
                        feature_1 = HaarFeature(FEATURE[i], (x, y), f_width, f_height, 0, 1)
                        feature_2 = HaarFeature(FEATURE[i], (x, y), f_width, f_height, 0, -1)
                        f_list.append(feature_1)
                        f_list.append(feature_2)
                        f_cnt += 1
        f_num[f_name] = f_cnt

    f_num['total'] = (len(f_list) / 2)
    return f_list, f_num

