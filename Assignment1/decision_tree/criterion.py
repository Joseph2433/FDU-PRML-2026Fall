"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    # =============== TODO (students) ===============
    n_samples = y.size
    if n_samples == 0:
        return 0.0

    # H(parent) - sum_child (n_child / n_parent) * H(child).
    for count in all_labels.values():
        probability = count / n_samples
        info_gain -= probability * math.log2(probability)
    for labels, child_size in ((left_labels, l_y.size),
                               (right_labels, r_y.size)):
        if child_size == 0:
            continue
        child_entropy = 0.0
        for count in labels.values():
            probability = count / child_size
            child_entropy -= probability * math.log2(probability)
        info_gain -= (child_size / n_samples) * child_entropy
    # ===============================================
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    # =============== TODO (students) ===============
    if y.size == 0:
        return 0.0
    split_info = 0.0
    for child_size in (l_y.size, r_y.size):
        if child_size == 0:
            continue
        proportion = child_size / y.size
        split_info -= proportion * math.log2(proportion)
    # An empty branch gives no real split and a zero denominator.
    if split_info == 0.0:
        return 0.0
    info_gain /= split_info
    # ===============================================
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # =============== TODO (students) ===============
    n_samples = y.size
    if n_samples == 0:
        return 0.0
    before = 1.0 - sum((count / n_samples) ** 2
                       for count in all_labels.values())
    for labels, child_size in ((left_labels, l_y.size),
                               (right_labels, r_y.size)):
        if child_size == 0:
            continue
        child_gini = 1.0 - sum((count / child_size) ** 2
                               for count in labels.values())
        after += (child_size / n_samples) * child_gini
    # ===============================================
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # =============== TODO (students) ===============
    n_samples = y.size
    if n_samples == 0:
        return 0.0
    before = 1.0 - max(all_labels.values()) / n_samples
    for labels, child_size in ((left_labels, l_y.size),
                               (right_labels, r_y.size)):
        if child_size == 0:
            continue
        child_error = 1.0 - max(labels.values()) / child_size
        after += (child_size / n_samples) * child_error
    # ===============================================
    return before - after
