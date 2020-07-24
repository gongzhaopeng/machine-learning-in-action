import operator

from numpy import *


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = tile(in_x, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(
        class_count.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    return sorted_class_count[0][0]


def file2matrix(file_name):
    fr = open(file_name)
    number_of_lines = len(fr.readlines())
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    fr = open(file_name)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_dataset = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_dataset = data_set - tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('../source-by-author/Ch02/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]):
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games? "))
    ff_miles = float(input("frequent flier miles earned per year? "))
    ice_cream = float(input("liters of ice cream consumed per year? "))
    dating_data_mat, dating_labels = file2matrix('../source-by-author/Ch02/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person: ", result_list[classifier_result - 1])


def img2vector(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect
