args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript run_anova_ancova.R <input.csv> <formula> <output.txt> <anova_type>")
}
input_file <- args[1]
formula_text <- args[2]
output_file <- args[3]
typ <- as.integer(args[4])

df <- read.csv(input_file, stringsAsFactors = TRUE)
sink(output_file)
cat("OLS + ANOVA table\n\n")
fit <- lm(as.formula(formula_text), data = df)
print(summary(fit))
cat("\nANOVA (car::Anova Type II/III)\n")
if (!requireNamespace("car", quietly = TRUE)) {
  cat("Package 'car' not installed; falling back to stats::anova.\n")
  print(anova(fit))
} else {
  print(car::Anova(fit, type = typ))
}
sink()
