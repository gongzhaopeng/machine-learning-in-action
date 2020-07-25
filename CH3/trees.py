from math import log


def calc_shannon_ent(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for featVec in dataset:
        current_label = featVec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for label in label_counts:
        prob = float(label_counts[label]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels
