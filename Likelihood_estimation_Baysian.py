# Define the likelihood function (binomial distribution for duplicates)
def likelihood(theta, n, k):
    # n: total number of rows
    # k: observed number of duplicates
    return binom.pmf(k, n, theta)

# Specify the prior distribution (e.g., Beta distribution)
# Assume a Beta(1, 1) prior (uniform prior between 0 and 1)
def prior(theta):
    return 1  # Uniform prior between 0 and 1 (constant)

# Compute the posterior distribution using Bayes' theorem
def posterior(theta, n, k):
    return likelihood(theta, n, k) * prior(theta)

# Estimate the probability of duplicates using posterior distribution
n_rows = len(df0)
theta_values = np.linspace(0, 1, 100)  # Range of theta values (probability of duplicates)
posterior_values = posterior(theta_values, n_rows, num_duplicates)

# Find the maximum likelihood estimate (MLE) of theta (probability of duplicates)
mle_theta = theta_values[np.argmax(posterior_values)]

print(f"Maximum Likelihood Estimate (MLE) of theta (probability of duplicates): {mle_theta:.4f}")
