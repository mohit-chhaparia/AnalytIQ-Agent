args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_linear_model.R <input.csv> <formula> <output.txt>")
}
input_file <- args[1]
formula_text <- args[2]
output_file <- args[3]

df <- read.csv(input_file, stringsAsFactors = TRUE)
sink(output_file)
cat("Linear Model (lm)\n\n")
fit <- lm(as.formula(formula_text), data = df)
print(summary(fit))
cat("\nAIC:\n")
print(AIC(fit))
sink()
