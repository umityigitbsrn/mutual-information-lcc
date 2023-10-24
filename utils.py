import numpy as np
import galois


# utils
def modulo_inverse(x, p):
    return pow(x, -1, p)


def finite_field_division(quotient, dividend, p):
    # find modulo inverse of dividend and multiply with the quotient
    mod_inv_div = modulo_inverse(dividend, p)
    return (mod_inv_div * quotient) % p


def poly_coefficients(evaluated_points, evaluation_points, p):
    vander_matrix = np.vander(evaluation_points, increasing=True) % p
    galois_field = galois.GF(p)
    vander_matrix = galois_field(vander_matrix)
    inv_vander_matrix = np.linalg.inv(vander_matrix)

    evaluated_points_in_field = galois_field(np.asarray(evaluated_points)[:, np.newaxis])

    return np.asarray([x for x in np.squeeze(inv_vander_matrix @ evaluated_points_in_field)])


def pol_mul_mod(first, second, prime):
    resulting_pol = np.zeros(2 * (len(first) - 1) + 1)
    for first_idx, first_el in enumerate(first):
        for second_idx, second_el in enumerate(second):
            resulting_pol[first_idx + second_idx] = (resulting_pol[first_idx + second_idx] + (
                    (first_el * second_el) % prime)) % prime
    return resulting_pol
