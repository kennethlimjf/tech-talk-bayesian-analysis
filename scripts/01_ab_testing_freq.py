"""
AB Testing: Frequentist way
===========================
Given 2 versions of web pages, A and B.

Page A has 500 visits and 110 click-through.
Page B has 1000 visits and 520 click-through.

We would like to know which version works better for conversion.
"""

from scipy.stats import ttest_ind, fisher_exact, bernoulli

TTEST_TEMPLATE = \
"""
Test results: %s
==============
statistic = %.2f
p-value = %.50f
"""


def generate_web_data(visits, mean):
    return bernoulli.rvs(mean, size=visits)


def two_tail_ttest(ctr_a, visits_a, ctr_b, visits_b):
    """ Performs a two-tailed T-test
    Null hypothesis: H_0 = H_1
    Alternative hypothesis: H_0 != H_1
    """
    p_a = ctr_a / visits_a
    p_b = ctr_b / visits_b

    data_a = generate_web_data(visits_a, p_a)
    data_b = generate_web_data(visits_b, p_b)

    ttest_result = ttest_ind(data_a, data_b, equal_var=False)

    print(TTEST_TEMPLATE % ("Two-tailed t-test",
        ttest_result.statistic, ttest_result.pvalue))


def one_tail_fisher_exact_test(ctr_a, visits_a, ctr_b, visits_b):
    """ Performs a one-tailed T-test
    Null hypothesis: H0 = H1
    Alternative hypothesis: H0 < H1

    Contingency table:
              |  A  |  B  |
    =======================
    Clicks    | 110 | 520 |
    -----------------------
    Non-clicks| 390 | 480 |
    =======================
    """
    non_ctr_a = visits_a - ctr_a
    non_ctr_b = visits_b - ctr_b

    oddsratio, pvalue = fisher_exact([[ctr_a, ctr_b], [non_ctr_a, non_ctr_b]],
        alternative="less")

    print(TTEST_TEMPLATE % ("One-sided fisher exact test",
        oddsratio, pvalue))



if __name__ == "__main__":
    ctr_a, visits_a = 110, 500
    ctr_b, visits_b = 520, 1000

    two_tail_ttest(ctr_a, visits_a, ctr_b, visits_b)
    one_tail_fisher_exact_test(ctr_a, visits_a, ctr_b, visits_b)
