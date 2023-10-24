import numpy as np
import math
import itertools


def count_occurrences(item, domain, X_idx, Y_idx):
    joint_occurrences = 0
    X_marginal_occurrences = 0
    Y_marginal_occurrences = 0
    item = list(item)
    for domain_item in domain:
        if (domain_item[(X_idx + Y_idx)] == item).all():
            joint_occurrences += 1

        if (domain_item[X_idx] == item[:len(X_idx)]).all():
            X_marginal_occurrences += 1

        if (domain_item[Y_idx] == item[len(X_idx):]).all():
            Y_marginal_occurrences += 1

    return joint_occurrences, X_marginal_occurrences, Y_marginal_occurrences


def calculate_mutual_information_item(item, domain, X_idx, Y_idx):
    joint_occurrences, X_marginal_occurrences, Y_marginal_occurrences = count_occurrences(item, domain, X_idx, Y_idx)
    joint_prob = joint_occurrences / len(domain)
    X_marginal_prob = X_marginal_occurrences / len(domain)
    Y_marginal_prob = Y_marginal_occurrences / len(domain)
    if joint_prob * X_marginal_prob * Y_marginal_prob != 0:
        return joint_prob * math.log(joint_prob / (X_marginal_prob * Y_marginal_prob))
    else:
        return 0


def calculate_mutual_information_domain(domain, X_idx, Y_idx, X_range_list, Y_range_list):
    inner_domain_range_list = []
    len_inner_domain = 1
    for X_range in X_range_list:
        len_inner_domain *= X_range
        inner_domain_range_list.append(np.arange(X_range))

    for Y_range in Y_range_list:
        len_inner_domain *= Y_range
        inner_domain_range_list.append(np.arange(Y_range))

    inner_domain_generator = itertools.product(*inner_domain_range_list)
    mutual_information = 0
    for idx, inner_domain_item in enumerate(inner_domain_generator):
        mutual_information += calculate_mutual_information_item(inner_domain_item, domain, X_idx, Y_idx)
        if (idx + 1) % 100 == 0:
            print('iteration {}/{}'.format(idx + 1, len_inner_domain))
    return mutual_information
