data {
  int<lower=1> N; // number of arms
  int<lower=1> D; // dimension of context vectors
  int<lower=1> T; // number of iterations
  int<lower=1> n; // batch size
  matrix[T, D] X; // context vectors
  int<lower=0, upper=1> y[T, N]; // binary rewards for each arm
}

parameters {
  vector[D] beta[N]; // linear regression coefficients for each arm
  vector<lower=0>[D] sigma[N]; // variance for each arm's coefficients
}

model {
  for (t in 1:T) {
    // sample beta from posterior distribution
    for (i in 1:N) {
      beta[i] ~ normal(beta[i], sigma[i]);
    }
    // compute estimated reward for each arm
    real theta[N];
    for (i in 1:N) {
      theta[i] = dot_product(X[t], beta[i]);
    }
    
    // select best arm from batch using Thompson Sampling
    int batch[n];
    for (j in 1:n) {
      real samples[N];
      for (i in 1:N) {
        samples[i] = normal_rng(theta[i], sigma[i]);
      }
      batch[j] = arg_max(samples);
    }
    int a = mode(batch);
    
    // update beta_hat for selected arm using observed reward
    beta_hat[a] = beta_hat[a] + y[t, a] * X[t];
  }
}
