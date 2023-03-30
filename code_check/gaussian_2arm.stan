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
  int<lower=0> N; // number of batches
  int<lower=0> batch_size; // number of trials in each batch
  vector[2] mu0; // prior means of the arms
  real<lower=0> sigma0; // prior standard deviation of the arms
}
parameters {
  vector[2] mu; // true means of the rewards
}
model {
  mu ~ normal(mu0, sigma0);
}

generated quantities {
  int chosen_arm[N * batch_size];
  vector[N * batch_size] rewards;
  vector[N] batch_allocation_prob;
  
  real current_prob;

  for (n in 1:N) {
    real batch_prob_sum = 0;
    for (b in 1:batch_size) {
      int idx = (n - 1) * batch_size + b;
      vector[2] sampled_means = to_vector(normal_rng(mu, sigma0));
      chosen_arm[idx] = arg_max(sampled_means);
      rewards[idx] = normal_rng(mu[chosen_arm[idx]], sigma0);

      // Compute allocation probabilities for each arm
      current_prob = exp(normal_lpdf(sampled_means[1] | mu[1], sigma0)) /
                                  (exp(normal_lpdf(sampled_means[1] | mu[1], sigma0)) + exp(normal_lpdf(sampled_means[2] | mu[2], sigma0)));
      batch_prob_sum += current_prob;
    }
    // Average allocation probability for the batch
    batch_allocation_prob[n] = batch_prob_sum / batch_size;
  }
}

