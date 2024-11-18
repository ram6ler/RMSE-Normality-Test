import math
import statistics
from scipy.stats import lognorm
from .data_types import Sample, Proportion, Population, Statistic
from .fits import log_normal_models


def random_sample(
    sample_size: int,
    population: Population,
) -> Sample:
    """
    Returns a random sample of size `sample_size` drawn from
    population `population`.
    """
    return sorted(population() for _ in range(sample_size))


def rmse_statistic(
    sample: list[float],
) -> tuple[list[float], list[float], list[float], float]:
    n = len(sample)
    m = statistics.mean(sample)
    s = statistics.stdev(sample)
    sorted_sample = sorted(sample)
    inv_cdf = statistics.NormalDist(0.0, 1.0).inv_cdf
    tzs = [inv_cdf((2 * k + 1) / (2 * n)) for k in range(n)]
    zs = [(x - m) / s for x in sorted_sample]
    errors = [tz - z for tz, z in zip(tzs, zs)]
    rmse = math.sqrt(sum(e**2 for e in errors) / (n - 1))
    return tzs, zs, errors, rmse


class RMSETest:
    """
    A formal test for normality based on an RMSE statistic associated
    with a QQ-plot generated for a sample drawn from a population with
    an unknown distribution.
    """

    def __init__(self, sample: Sample):
        self._sample = sorted(sample)
        self._n = len(sample)
        (
            self._theoretical_z_scores,
            self._z_scores,
            self._errors,
            self._rmse,
        ) = rmse_statistic(self._sample)

        if self._n in log_normal_models:
            mu, sigma = log_normal_models[self._n]
            model = lognorm(s=sigma, scale=math.exp(mu))
            self._p_value = 1 - float(model.cdf(self._rmse))
        else:
            raise Exception

    @property
    def n(self) -> int:
        """
        The size of the sample.
        """
        return self._n

    @property
    def sample(self) -> Sample:
        """
        The sorted values in the sample.
        """
        return [x for x in self._sample]

    @property
    def z_scores(self) -> Sample:
        """
        The number of standard deviations each point in the sample
        is from the mean.
        """
        return [z for z in self._z_scores]

    @property
    def theoretical_z_scores(self) -> Sample:
        """
        The expected number of standard deviations each point in the
        sample is from the mean under the hypothesis that the population
        the sample was drawn from has a normal distribution.
        """
        return [z for z in self._theoretical_z_scores]

    @property
    def rmse(self) -> Statistic:
        """
        The value of the RMSE statistic associated with this sample.

        This is the root of the mean squared error associated with the
        points in the QQ-plot if we treat the theoretical z-scores as
        predictions for the z-scores; a denominator of n - 1 is used
        to address bias in using the sample statistic to estimate the
        population statistic, analogous to how the population's standard
        deviation is estimated from a sample.
        """
        return self._rmse

    @property
    def p_value(self) -> Proportion:
        """
        The estimated proportion of values in the RMSE statistic sampling
        distribution more extreme than the RMSE statistica associated with
        this sample.
        """
        return self._p_value

    def critical_rmse(self, significance_level: Proportion) -> Statistic:
        """
        The estimated critical RMSE statistic value for the given significance
        level for a sample of this size. Typical significance levels are 0.01,
        0.05 or 0.10.
        """
        if self._n in log_normal_models:
            mu, sigma = log_normal_models[self._n]
            model = lognorm(s=sigma, scale=math.exp(mu))
            return model.ppf(1 - significance_level)
        raise Exception

    def quantile(self, proportion: Proportion) -> Statistic:
        """
        The RMSE statistic value that acts as an upper boundary for a proportion
        of values in the RMSE statistic sampling distribution.
        """
        if self._n in log_normal_models:
            mu, sigma = log_normal_models[self._n]
            model = lognorm(s=sigma, scale=math.exp(mu))
            return model.ppf(proportion)
        raise Exception

    def __str__(self) -> str:
        width = 10

        result = """
Key:
  x: sorted sample data                           
 tz: expected z-scores under normality hypothesis 
  z: actual sample z-scores                       
  e: errors in predicting z with tz   

Data:
"""

        def row(x="", tz="", z="", e="", j="|", f=" ") -> str:
            return j + j.join(v.center(width, f) for v in (x, tz, z, e)) + j + "\n"

        result += (
            row(j=".", f="-") + row(x="x", tz="tz", z="z", e="e") + row(j=":", f="-")
        )
        for k in range(self._n):
            x = self._sample[k]
            tz = self._theoretical_z_scores[k]
            z = self._z_scores[k]
            e = self._errors[k]
            result += row(
                x=(" " if tz >= 0 else "") + f"{x:.2f}",
                tz=(" " if tz >= 0 else "") + f"{tz:.3f}",
                z=(" " if z >= 0 else "") + f"{z:.3f}",
                e=(" " if e >= 0 else "") + f"{e:.3f}",
            )
        result += row(j="'", f="-")
        result += f"""
Results:
    n: {self.n}
 RMSE: {self.rmse:.3f}
    p: {self.p_value:.3f}

"""

        return result

    def __repr__(self) -> str:
        return str(self)
