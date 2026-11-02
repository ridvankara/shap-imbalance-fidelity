# dgp.R - Veri Üetim Sü
# SHAP Stability Under Class Imbalance Treatments

library(MASS)
library(Matrix)
library(pROC)

# Korelasyon matrisi
build_sigma <- function(p, corr_type, block_size = 5, rho = 0.6) {
  if (corr_type == "independent") {
    return(diag(p))
  }
  if (corr_type == "block") {
    n_blocks <- p / block_size
    blocks <- replicate(
      n_blocks,
      matrix(rho, block_size, block_size),
      simplify = FALSE
    )
    Sigma <- as.matrix(Matrix::bdiag(blocks))
    diag(Sigma) <- 1
    return(Sigma)
  }
  stop("corr_type must be \"independent\" or \"block\"")
}

# Beta vektö(yapısal, öklendirilmemiş)
build_beta <- function(p, corr_type) {
  beta <- numeric(p)
  if (corr_type == "independent") {
    beta[1:3] <- 1.0
    beta[4:5] <- 0.5
    beta[6:7] <- 0.25
  } else if (corr_type == "block") {
    beta[1:3] <- 1.0
    beta[4:5] <- 0.5
    beta[6:7] <- 0.25
  }
  beta
}

# Intercept cozucu
solve_intercept <- function(beta, Sigma, target_prevalence, n_solve = 1e5) {
  X <- mvrnorm(n_solve, mu = rep(0, length(beta)), Sigma = Sigma)
  eta_no_intercept <- as.numeric(X %*% beta)
  f <- function(alpha) {
    mean(plogis(alpha + eta_no_intercept)) - target_prevalence
  }
  uniroot(f, interval = c(-15, 15))$root
}

# Bayes-optimal AUC
bayes_auc <- function(beta, Sigma, alpha, n_test = 1e5) {
  X <- mvrnorm(n_test, mu = rep(0, length(beta)), Sigma = Sigma)
  eta <- alpha + as.numeric(X %*% beta)
  p_y <- plogis(eta)
  y <- rbinom(n_test, 1, p_y)
  as.numeric(pROC::roc(y, p_y, quiet = TRUE)$auc)
}

# Beta olcek kalibrasyonu
calibrate_beta_scale <- function(beta, Sigma, prevalence,
                                  target_auc = 0.82,
                                  n_test = 1e5,
                                  seed = 42) {
  set.seed(seed)
  X <- mvrnorm(n_test, mu = rep(0, length(beta)), Sigma = Sigma)
  u <- runif(n_test)
  
  f <- function(s) {
    beta_s <- s * beta
    alpha <- solve_intercept(beta_s, Sigma, prevalence)
    eta <- alpha + as.numeric(X %*% beta_s)
    p_y <- plogis(eta)
    y <- as.integer(u < p_y)
    if (length(unique(y)) < 2) return(NA)
    auc <- as.numeric(pROC::roc(y, p_y, quiet = TRUE)$auc)
    auc - target_auc
  }
  uniroot(f, interval = c(0.05, 2.0), tol = 1e-4)$root
}

# Veri ureteci
generate_data <- function(n, beta, alpha, Sigma, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X <- mvrnorm(n, mu = rep(0, length(beta)), Sigma = Sigma)
  eta <- alpha + as.numeric(X %*% beta)
  p_y <- plogis(eta)
  y <- rbinom(n, 1, p_y)
  colnames(X) <- paste0("x", seq_len(length(beta)))
  list(X = X, y = y)
}

