import math
import statistics
from .rmse_test import RMSETest, rmse_statistic, random_sample
from .populations import Populations
from .data_types import Sample, Statistic, Proportion, Population


def monte_carlo_simulations(
    sample_size: int,
    simulations=10_000,
) -> Sample:
    """
    Returns simulated RMSE values under the normality hypothesis.
    """
    return [
        rmse_statistic(
            random_sample(
                sample_size,
                Populations.NORMAL(0.0, 1.0),
            )
        )[-1]
        for _ in range(simulations)
    ]


def log_normal_fit(rmse_values: Sample) -> tuple[Statistic, Statistic]:
    """
    Returns estimated parameter values for a log-normal fit of simulated RMSE values.
    """
    mu = statistics.mean(math.log(x) for x in rmse_values)
    sigma = statistics.stdev(math.log(x) for x in rmse_values)
    return mu, sigma


def rmse_power_estimate(
    sample_size: int,
    population: Population,
    experiments=10_000,
    significance_level=0.05,
) -> Proportion:
    """
    Uses Monte Carlo simulations to estimate the power of the RMSE test
    in the context of a given population, sample size and significance level.

    Example:

    ```python
    from rmse_test import Populations, rmse_power_estimate

    Populations.seed(0)

    print(
        rmse_power_estimate(
            sample_size=15,
            population=Populations.LOG_NORMAL(0.0, 1.0),
            experiments=100_000,
            significance_level=0.05,
        )
    )
    ```
    """
    successes = 0
    for _ in range(experiments):
        sample = random_sample(sample_size, population)
        test = RMSETest(sample)
        if test.rmse > test.critical_rmse(significance_level):
            successes += 1

    return successes / experiments
