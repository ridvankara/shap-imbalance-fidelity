# balancing.R - Dengeleme Stratejileri

library(themis)

apply_balancing <- function(X, y, strategy, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  feature_names <- colnames(X)
  if (is.null(feature_names)) {
    feature_names <- paste0("x", seq_len(ncol(X)))
    colnames(X) <- feature_names
  }
  
  result <- switch(strategy,
    "none" = list(X = X, y = y, weights = rep(1, length(y))),
    
    "random_over" = {
      pos_idx <- which(y == 1)
      neg_idx <- which(y == 0)
      n_target <- length(neg_idx)
      pos_sampled <- sample(pos_idx, n_target, replace = TRUE)
      keep <- c(neg_idx, pos_sampled)
      list(X = X[keep, , drop = FALSE], 
           y = y[keep], 
           weights = rep(1, length(keep)))
    },
    
    "random_under" = {
      pos_idx <- which(y == 1)
      neg_idx <- which(y == 0)
      n_target <- length(pos_idx)
      neg_sampled <- sample(neg_idx, n_target, replace = FALSE)
      keep <- c(pos_idx, neg_sampled)
      list(X = X[keep, , drop = FALSE], 
           y = y[keep], 
           weights = rep(1, length(keep)))
    },
    
    "smote" = {
      df <- data.frame(X, .y = factor(y, levels = c(0, 1)))
      rec <- themis::smote(df, var = ".y", over_ratio = 1)
      X_new <- as.matrix(rec[, feature_names])
      y_new <- as.integer(as.character(rec$.y))
      list(X = X_new, y = y_new, weights = rep(1, nrow(X_new)))
    },
    
    "adasyn" = {
      df <- data.frame(X, .y = factor(y, levels = c(0, 1)))
      rec <- themis::adasyn(df, var = ".y", over_ratio = 1)
      X_new <- as.matrix(rec[, feature_names])
      y_new <- as.integer(as.character(rec$.y))
      list(X = X_new, y = y_new, weights = rep(1, nrow(X_new)))
    },
    
    "class_weight" = {
      n_neg <- sum(y == 0)
      n_pos <- sum(y == 1)
      w <- ifelse(y == 1, n_neg / n_pos, 1)
      list(X = X, y = y, weights = w)
    },
    
    stop("Unknown strategy: ", strategy)
  )
  result
}

