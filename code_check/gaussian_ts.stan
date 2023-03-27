functions { // writing arg_max by ourselves
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
  int<lower=0> K; // number of arms
  int<lower=0> N; // number of trials
  vector[K] mu0; // prior means of the arms
  real<lower=0> sigma0; // prior standard deviation of the arms
  real<lower=0> sigma; // true standard deviation of the rewards
}
parameters {
  vector[K] mu; // true means of the rewards
}
model {
  mu ~ normal(mu0, sigma0);
}
generated quantities {
  int chosen_arm[N];
  vector[N] rewards;

  for (n in 1:N) {
    vector[K] sampled_means = to_vector(normal_rng(mu, sigma0));
    chosen_arm[n] = arg_max(sampled_means);
    rewards[n] = normal_rng(mu[chosen_arm[n]], sigma);
  }
}
