args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript diagnostics.R <input.csv> <formula> <output.txt>")
}
input_file <- args[1]
formula_text <- args[2]
output_file <- args[3]

df <- read.csv(input_file, stringsAsFactors = TRUE)
sink(output_file)
cat("Basic OLS diagnostics\n\n")
fit <- lm(as.formula(formula_text), data = df)
cat("Residual summary:\n")
print(summary(residuals(fit)))
cat("\nShapiro-Wilk normality test on residuals (may fail for large n):\n")
if (length(residuals(fit)) <= 5000) {
  print(shapiro.test(residuals(fit)))
} else {
  cat("Skipped: sample too large for Shapiro-Wilk default.\n")
}
sink()
