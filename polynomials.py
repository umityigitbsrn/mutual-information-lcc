import numpy as np

from utils import poly_coefficients, finite_field_division, pol_mul_mod


# lagrange coded computing polynomial
class Poly(object):
    def __init__(self, coefficients, p):
        self.__coefficients = coefficients
        self.degree = len(coefficients)
        self.p = p

    @property
    def coefficients(self):
        return self.__coefficients

    def __call__(self, evaluation_point, *args, **kwargs):
        return np.sum(
            np.asarray([evaluation_point ** i for i in range(len(self.__coefficients))]) * self.__coefficients) % self.p

    def __mul__(self, other):
        return Poly(pol_mul_mod(self.__coefficients, other.coefficients, self.p), self.p)


class LCCPoly(object):
    def __init__(self, beta_arr, secret_arr, K, T, p):
        self.K = K
        self.T = T
        self.p = p
        self.beta_arr = beta_arr
        self.secret_arr = secret_arr
        self.degree = K + T - 1

        self._random_arr = []
        self.__fill_random_arr()

        self.__coefficients = poly_coefficients(secret_arr + self._random_arr, beta_arr, p)

    def __fill_random_arr(self):
        for j in range(self.K, self.K + self.T):
            self._random_arr.append(np.random.randint(0, high=self.p))

    @property
    def random_arr(self):
        return self._random_arr

    @property
    def coefficients(self):
        return self.__coefficients

    def __call__(self, evaluation_point, *args, **kwargs):
        result = 0
        for j in range(self.K):
            curr_secret = self.secret_arr[j]
            curr_mul = 1
            for k in range(self.K + self.T):
                if k == j:
                    continue
                else:
                    first = (evaluation_point - self.beta_arr[k]) % self.p
                    second = (self.beta_arr[j] - self.beta_arr[k]) % self.p
                    curr_mul = (curr_mul * finite_field_division(first, second, self.p)) % self.p
            result = (result + ((curr_secret * curr_mul) % self.p)) % self.p
        for j in range(self.K, self.K + self.T):
            curr_random = self._random_arr[j - self.K]
            curr_mul = 1
            for k in range(self.K + self.T):
                if k == j:
                    continue
                else:
                    first = (evaluation_point - self.beta_arr[k]) % self.p
                    second = (self.beta_arr[j] - self.beta_arr[k]) % self.p
                    curr_mul = (curr_mul * finite_field_division(first, second, self.p)) % self.p
            result = (result + ((curr_random * curr_mul) % self.p)) % self.p
        return result

    def __mul__(self, other):
        return Poly(pol_mul_mod(self.__coefficients, other.coefficients, self.p), self.p)


class InterpolatedPoly(object):
    def __init__(self, evaluated_points, evaluation_points, p):
        self.evaluated_points = evaluated_points
        self.evaluation_points = evaluation_points
        self.p = p
        self.degree = len(evaluated_points) - 1

        self.__coefficients = poly_coefficients(evaluated_points, evaluation_points, p)

    @property
    def coefficients(self):
        return self.__coefficients

    def __call__(self, evaluation_point, *args, **kwargs):
        result = 0
        for j in range(len(self.evaluated_points)):
            curr_secret = self.evaluated_points[j]
            curr_mul = 1
            for k in range(len(self.evaluated_points)):
                if k == j:
                    continue
                else:
                    first = (evaluation_point - self.evaluation_points[k]) % self.p
                    second = (self.evaluation_points[j] - self.evaluation_points[k]) % self.p
                    curr_mul = (curr_mul * finite_field_division(first, second, self.p)) % self.p
            result = (result + ((curr_secret * curr_mul) % self.p)) % self.p
        return result

    def __mul__(self, other):
        return Poly(pol_mul_mod(self.__coefficients, other.coefficients, self.p), self.p)
