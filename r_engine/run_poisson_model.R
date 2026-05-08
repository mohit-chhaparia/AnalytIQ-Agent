args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_poisson_model.R <input.csv> <formula> <output.txt>")
}
input_file <- args[1]
formula_text <- args[2]
output_file <- args[3]

df <- read.csv(input_file, stringsAsFactors = TRUE)
sink(output_file)
cat("Poisson GLM\n\n")
fit <- glm(as.formula(formula_text), data = df, family = poisson())
print(summary(fit))
cat("\nAIC:\n")
print(AIC(fit))
cat("\nDispersion (deviance/df.residual):\n")
print(if (fit$df.residual > 0) fit$deviance / fit$df.residual else NA)
sink()
