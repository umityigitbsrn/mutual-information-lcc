from polynomials import InterpolatedPoly
import numpy as np
import itertools


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
