functions {
  int arg_max(vector x) {
    int idx = 1;
    for (i in 2:num_elements(x)) {
      if (x[i] > x[idx]) {
        idx = i;
      }
    }
    return idx;
  }
}

data {
  int<lower=0> N; // number of trials
  vector[2] mu0; // prior means of the arms
  real<lower=0> sigma0; // prior standard deviation of the arms
  real<lower=0> sigma; // true standard deviation of the rewards
}
parameters {
  vector[2] mu; // true means of the rewards
}
model {
  mu ~ normal(mu0, sigma0);
}
generated quantities {
  int chosen_arm[N];
  vector[N] rewards;
  vector[N] allocation_prob;

  for (n in 1:N) {
    vector[2] sampled_means = to_vector(normal_rng(mu, sigma0));
    chosen_arm[n] = arg_max(sampled_means);
    rewards[n] = normal_rng(mu[chosen_arm[n]], sigma);

    // Compute allocation probabilities for each arm
    allocation_prob[n] = exp(normal_lpdf(sampled_means[1] | mu[1], sigma0)) /
                              (exp(normal_lpdf(sampled_means[1] | mu[1], sigma0)) + exp(normal_lpdf(sampled_means[2] | mu[2], sigma0)));
  }
}
