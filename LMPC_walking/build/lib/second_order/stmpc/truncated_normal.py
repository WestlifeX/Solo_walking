# headers
from numpy import zeros, array, random, mean, std, var
from scipy.stats import norm
# ------------------------------------------------------------------------------
#                           Truncated PDF
# ------------------------------------------------------------------------------
def truncated_normal_pdf(x, mu_hat, sigma_hat, lb, ub):
    CDF_lb = norm.cdf(lb, loc=mu_hat, scale=sigma_hat)
    CDF_ub = norm.cdf(ub, loc=mu_hat, scale=sigma_hat)
    normalizer = 1/(CDF_ub-CDF_lb)
    truncated_pdf = normalizer*norm.pdf(x, loc=mu_hat, scale=sigma_hat)
    return truncated_pdf
# ------------------------------------------------------------------------------
#                           Truncated CDF
# ------------------------------------------------------------------------------
def truncated_normal_cdf(x, mu_hat, sigma_hat, lb, ub):
    CDF_lb = norm.cdf(lb, loc=mu_hat, scale=sigma_hat)
    CDF_ub = norm.cdf(ub, loc=mu_hat, scale=sigma_hat)
    CDF_x  = norm.cdf(x , loc=mu_hat, scale=sigma_hat)
    normalizer = 1/(CDF_ub-CDF_lb)
    if (x <= lb):
        truncated_cdf = 0.0
    elif (x < ub):
        truncated_cdf = normalizer*(CDF_x - CDF_lb)
    else:
        truncated_cdf = 1.0
    return truncated_cdf
# ------------------------------------------------------------------------------
#                          inverse Truncated CDF
# ------------------------------------------------------------------------------
def truncated_normal_inverse_cdf(p, mu_hat, sigma_hat, lb, ub):
    alpha = (lb-mu_hat)/sigma_hat
    beta  = (ub-mu_hat)/sigma_hat
    alpha_cdf = norm.cdf(alpha)
    beta_cdf  = norm.cdf(beta)
    xi_cdf = p*(beta_cdf - alpha_cdf) + alpha_cdf
    xi = norm.ppf(xi_cdf)
    x = mu_hat + (sigma_hat*xi)
    return x
# ------------------------------------------------------------------------------
#              sample from a truncated normal distribution
# ------------------------------------------------------------------------------
def sample_from_truncated_normal(mu_hat, sigma_hat, lb, ub):
    p = random.uniform()
    x = truncated_normal_inverse_cdf(p, mu_hat, sigma_hat, lb, ub)
    return x

# ------------------------------------------------------------------------------
#         compute the statistics of a truncated normal distribution
# ------------------------------------------------------------------------------
def compute_mean_truncated_normal(mu_hat, sigma_hat, lb, ub):
    alpha = (lb-mu_hat)/sigma_hat
    beta  = (ub-mu_hat)/sigma_hat
    alpha_pdf = norm.pdf(alpha)
    beta_pdf  = norm.pdf(beta)
    alpha_cdf = norm.cdf(alpha)
    beta_cdf  = norm.cdf(beta)
    CDF_lb = norm.cdf(lb, loc=mu_hat, scale=sigma_hat)
    CDF_ub = norm.cdf(ub, loc=mu_hat, scale=sigma_hat)
    normalizer = 1/(CDF_ub-CDF_lb)
    mu = mu_hat + sigma_hat*normalizer*(alpha_pdf-beta_pdf)
    return mu

def compute_variance_truncated_normal(mu_hat, sigma_hat, lb, ub):
    alpha = (lb-mu_hat)/sigma_hat
    beta  = (ub-mu_hat)/sigma_hat
    alpha_pdf = norm.pdf(alpha)
    beta_pdf  = norm.pdf(beta)
    alpha_cdf = norm.cdf(alpha)
    beta_cdf  = norm.cdf(beta)
    CDF_lb = norm.cdf(lb, loc=mu_hat, scale=sigma_hat)
    CDF_ub = norm.cdf(ub, loc=mu_hat, scale=sigma_hat)
    normalizer = 1/(CDF_ub-CDF_lb)
    variance  = (sigma_hat**2)*(1.0 \
                + (normalizer*(alpha*alpha_pdf - beta*beta_pdf)) \
                - (normalizer*(alpha_pdf - beta_pdf)** 2))
    return variance
# ------------------------------------------------------------------------------
#                            UNIT TESTS
# ------------------------------------------------------------------------------
if __name__=='__main__':
    print('unit tests for truncated normal distribution'.center(60,'*'))
    print('\n')
    no_samples = 10
    mu_hat     = 100.0  # original gaussian mean
    sigma_hat  = 25.0   # original gaussian std
    lb = 50.0           # lower bound of truncation
    ub = 150.0          # upper bound of truncation
    # 10 random lb <x< ub
    x = array([81.63, 137.962, 122.367, 103.704, 94.899,
               65.8326, 84.5743, 71.5672, 62.0654, 108.155])

    # compute the CDF of 10 samples and the inverse of those CDF to get back
    # the same samples
    truncated_cdf = zeros(no_samples)
    truncated_inverse_cdf = zeros(no_samples)
    for i in range(no_samples):
        truncated_cdf[i] = truncated_normal_cdf(x[i], mu_hat, sigma_hat, lb, ub)
        truncated_inverse_cdf[i] = truncated_normal_inverse_cdf(truncated_cdf[i],
                                    mu_hat, sigma_hat, lb, ub)
    print("x = ", x, '\n')
    print("truncated normal CDF = ", '\n')
    print(truncated_cdf, '\n')
    print("truncated normal inverse CDF = ", '\n')
    print(truncated_inverse_cdf, '\n')

    # compare sampled mean and variance with computed mean and variance
    no_samples = 10000
    samples = zeros(no_samples)
    for i in range(no_samples):
        samples[i] = sample_from_truncated_normal(mu_hat, sigma_hat, lb, ub)
    mu  = mean(samples)
    variance = var(samples)
    print("mu of sampled data  = ", mu)
    print("variance of sampled data = ", variance, '\n')
    computed_mu = compute_mean_truncated_normal(mu_hat, sigma_hat, lb, ub)
    computed_variance = compute_variance_truncated_normal(mu_hat, sigma_hat,
                                                          lb, ub)
    print("computed mu = ", computed_mu)
    print("computed variance = ", computed_variance)
