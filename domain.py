from polynomials import InterpolatedPoly
import numpy as np
import itertools
from utils import pol_mul_mod


def create_lcc_domain(K, data_range, T, prime, beta_arr):
    data_range_domain = np.arange(data_range)
    random_range_domain = np.arange(prime)
    data_range_domain = [data_range_domain for _ in range(K)]
    random_range_domain = [random_range_domain for _ in range(T)]
    secret_random_domain_generator = itertools.product(*data_range_domain, *random_range_domain, *data_range_domain,
                                                       *random_range_domain)

    domain_size = ((data_range ** K) * (prime ** T)) ** 2
    domain_feature_size = 3 * (K + T) + 2 * (K + T - 1) + 1
    domain = np.empty((domain_size, domain_feature_size))

    for idx, secret_random in enumerate(secret_random_domain_generator):
        secret_random = list(secret_random)
        domain[idx, :(2 * (K + T))] = secret_random
        first_interpolated_poly = InterpolatedPoly(secret_random[:(K + T)], beta_arr, prime)
        second_interpolated_poly = InterpolatedPoly(secret_random[(K + T):], beta_arr, prime)
        multiplied_poly = first_interpolated_poly * second_interpolated_poly
        for beta_idx, beta in enumerate(beta_arr):
            domain[idx, (2 * (K + T)) + beta_idx] = multiplied_poly(beta)

        domain[idx, (3 * (K + T)):] = multiplied_poly.coefficients

    return domain


def create_sss_dataset(prime, data_domain_range, num_of_samples, degree, norm=False):
    # creating the dataset
    ## starting from basic shamir secret sharing with real numbers
    ## creating the polynomials first (reverse order - higher degree in the beginning)
    first_poly_data = np.random.randint(data_domain_range, size=(num_of_samples, 1))
    first_poly_coeff = np.random.randint(prime, size=(num_of_samples, degree))
    first_poly_dataset = np.hstack([first_poly_data, first_poly_coeff])

    # to create a resulting polynomial without any zero in the largest degree
    second_poly_data = np.random.randint(data_domain_range, size=(num_of_samples, 1))
    second_poly_coeff = np.random.randint(prime, size=(num_of_samples, degree))
    second_poly_dataset = np.hstack([second_poly_data, second_poly_coeff])

    multiplied_poly_dataset = np.empty((num_of_samples, 2 * degree + 1))
    for idx in range(num_of_samples):
        multiplied_poly_dataset[idx] = pol_mul_mod(first_poly_dataset[idx], second_poly_dataset[idx], prime)

    mutual_information_dataset = np.empty((num_of_samples, 4 * degree + 3))
    mutual_information_dataset[:, (2 * degree + 2):] = multiplied_poly_dataset
    mutual_information_dataset[:, (degree + 1):(2 * degree + 2)] = second_poly_dataset
    mutual_information_dataset[:, :(degree + 1)] = first_poly_dataset
    mutual_information_dataset = mutual_information_dataset.astype(np.float64)

    if norm:
        mean_mi_dataset = np.mean(mutual_information_dataset)
        std_mi_dataset = np.std(mutual_information_dataset)
        mutual_information_dataset = (mutual_information_dataset - mean_mi_dataset) / std_mi_dataset

    return mutual_information_dataset
