#!/usr/bin/env Rscript
# Run ANOVA / ANCOVA for factorial/DOE-style designs.
# Usage: Rscript run_anova_ancova.R <input_csv> <formula> <output_txt>
#
# Required packages: car, emmeans, broom, tidyverse

suppressPackageStartupMessages({
  library(car)
  library(emmeans)
  library(broom)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript run_anova_ancova.R <input_csv> <formula> <output_txt>\n")
  quit(status = 1)
}

input_file  <- args[1]
formula_str <- args[2]
output_file <- args[3]

df <- read.csv(input_file)

# Auto-convert character columns to factors
df <- df %>% mutate(across(where(is.character), as.factor))

sink(output_file)

cat("=== ANOVA / ANCOVA ANALYSIS ===\n\n")
cat("Formula:", formula_str, "\n\n")

tryCatch({
  model <- lm(as.formula(formula_str), data = df)

  cat("--- Model Summary ---\n")
  print(summary(model))

  cat("\n--- Type III ANOVA Table (car::Anova) ---\n")
  print(Anova(model, type = "III"))

  cat("\n--- Tidy Coefficients (broom) ---\n")
  print(tidy(model))

  cat("\n--- Model Fit ---\n")
  cat("R-squared:    ", round(summary(model)$r.squared, 4), "\n")
  cat("Adj R-squared:", round(summary(model)$adj.r.squared, 4), "\n")
  cat("AIC:          ", round(AIC(model), 2), "\n")
  cat("BIC:          ", round(BIC(model), 2), "\n")

  cat("\n--- Diagnostics ---\n")
  shapiro_result <- shapiro.test(resid(model))
  cat("Shapiro-Wilk normality test on residuals:\n")
  cat("  W =", round(shapiro_result$statistic, 4),
      "  p-value =", round(shapiro_result$p.value, 4), "\n")
  if (shapiro_result$p.value < 0.05) {
    cat("  WARNING: Residuals may not be normally distributed (p < 0.05).\n")
  } else {
    cat("  Residuals appear normally distributed (p >= 0.05).\n")
  }

  cat("\n--- Levene's Test (Homogeneity of Variance) ---\n")
  tryCatch({
    # Extract first factor from formula
    fmla_terms <- all.vars(as.formula(formula_str))
    response_var <- fmla_terms[1]
    factor_var   <- fmla_terms[2]
    if (is.factor(df[[factor_var]]) || is.character(df[[factor_var]])) {
      levene_result <- leveneTest(df[[response_var]] ~ df[[factor_var]])
      print(levene_result)
    } else {
      cat("  Skipped: first predictor is not a factor.\n")
    }
  }, error = function(e) {
    cat("  Levene's test skipped:", conditionMessage(e), "\n")
  })

  cat("\n--- Estimated Marginal Means (emmeans) ---\n")
  tryCatch({
    fmla_terms <- all.vars(as.formula(formula_str))
    factor_vars <- fmla_terms[-1]
    if (length(factor_vars) > 0) {
      emm <- emmeans(model, as.formula(paste("~", factor_vars[1])))
      print(emm)
      cat("\nPairwise contrasts:\n")
      print(contrast(emm, method = "pairwise", adjust = "tukey"))
    }
  }, error = function(e) {
    cat("  emmeans skipped:", conditionMessage(e), "\n")
  })

}, error = function(e) {
  cat("ERROR fitting model:", conditionMessage(e), "\n")
})

sink()
cat("Output written to:", output_file, "\n")
