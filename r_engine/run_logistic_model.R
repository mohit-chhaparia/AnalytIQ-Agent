args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_logistic_model.R <input.csv> <formula> <output.txt>")
}
input_file <- args[1]
formula_text <- args[2]
output_file <- args[3]

df <- read.csv(input_file, stringsAsFactors = TRUE)
sink(output_file)
cat("Logistic Regression (glm binomial)\n\n")
fit <- glm(as.formula(formula_text), data = df, family = binomial())
print(summary(fit))
cat("\nAIC:\n")
print(AIC(fit))
sink()
