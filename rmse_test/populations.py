from typing import Callable
import numpy.random as random


class Populations:
    """
    A selection of random number generators.

    Example:

    ```python
    f = Populations.NORMAL(mu=0.0, sigma=1.0)
    print(f())
    ```
    """

    @staticmethod
    def seed(s: int) -> None:
        random.seed(s)

    @staticmethod
    def NORMAL(
        mu: float,
        sigma: float,
    ) -> Callable[[], float]:
        return lambda: random.normal(mu, sigma)

    @staticmethod
    def UNIFORM(
        a: float,
        b: float,
    ) -> Callable[[], float]:
        return lambda: random.uniform(a, b)

    @staticmethod
    def WEIBULL(
        a: float,
    ) -> Callable[[], float]:
        return lambda: random.weibull(a)

    @staticmethod
    def GAMMA(
        alpha: float,
        beta: float,
    ) -> Callable[[], float]:
        return lambda: random.gamma(alpha, beta)

    @staticmethod
    def TRIANGULAR(
        a: float,
        b: float,
    ) -> Callable[[], float]:
        return lambda: random.triangular(a, (a + b) / 2, b)

    @staticmethod
    def LOG_NORMAL(
        mu: float,
        sigma: float,
    ) -> Callable[[], float]:
        return lambda: random.lognormal(mu, sigma)

    @staticmethod
    def EXPONENTIAL(
        scale: float,
    ) -> Callable[[], float]:
        return lambda: random.exponential(scale)

    @staticmethod
    def BETA(
        alpha: float,
        beta: float,
    ) -> Callable[[], float]:
        return lambda: random.beta(alpha, beta)

    @staticmethod
    def CHI_SQUARED(
        df: float,
    ) -> Callable[[], float]:
        return lambda: random.chisquare(df)

    @staticmethod
    def T(
        df: float,
    ) -> Callable[[], float]:
        return lambda: random.standard_t(df)

    @staticmethod
    def PARETO(
        shape: float,
    ) -> Callable[[], float]:
        return lambda: random.pareto(shape)

    def __init__(self):
        raise NotImplementedError
