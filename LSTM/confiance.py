#########################
# Used to calculate the confiance of a prediction.
import scipy.stats


mu_1 = 4
mu_2 = 0

sigma_1 = 4.89452316190757
sigma_2 = 19.95029171612384

dist_1 = scipy.stats.norm(mu_1, sigma_1)
dist_2 = scipy.stats.norm(mu_2, sigma_2)

cdf_1_0 = dist_1.cdf(0)
cdf_2_0 = 1/2

def pdf(x) :
    pdf_1 = dist_1.pdf(x) / (2*(1-cdf_1_0))
    pdf_2 = dist_2.pdf(x)

    return pdf_1 + pdf_2


def cdf(x) :
    cdf_1 = (dist_1.cdf(x)-cdf_1_0) / (2*(1-cdf_1_0))
    cdf_2 = dist_2.cdf(x) - cdf_2_0

    return cdf_1 + cdf_2


def conf(x) :
    return 1 - cdf(x)