# Cumulative-link mixed models (R package 'ordinal').
#
# Usage:
#   Rscript clmm.R <human_long.csv> <human_out.csv> [<llm_long.csv> <llm_out.csv>]
#
# 1. Human: per task, rd ~ z(bc) + (1 | participant), logit link, 10-point adaptive
#    Gauss-Hermite quadrature; z(bc) standardizes B_C within the task. Wald 95% intervals.
#    The participant-factor level order affects estimates only at optimizer tolerance (~1e-5).
# 2. Optional LLM ensemble reference for the grouped descriptive figure: per task,
#    rd ~ x + (1 | model) with x = bc / 100 (Laplace approximation, package defaults).
#    Writes beta and the ordered thresholds in long format.

suppressPackageStartupMessages(library(ordinal))

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(2, 4))) {
  stop("Usage: Rscript clmm.R <human_long.csv> <human_out.csv> [<llm_long.csv> <llm_out.csv>]")
}
tasks <- c("DNC", "CTD", "FIP")

human <- read.csv(args[[1]], stringsAsFactors = FALSE)
human_rows <- lapply(tasks, function(task) {
  d <- human[human$task == task, ]
  d$participant <- factor(d$participant)
  d$rd <- ordered(d$rd, levels = 1:5)
  d$bc_z <- as.numeric(scale(d$bc))
  fit <- clmm(rd ~ bc_z + (1 | participant), data = d, link = "logit", Hess = TRUE, nAGQ = 10,
              control = clmm.control(maxIter = 200, maxLineIter = 200))
  est <- coef(summary(fit))["bc_z", ]
  data.frame(
    task = task,
    n_participants = nlevels(d$participant),
    n_trials = nrow(d),
    beta_z = est[["Estimate"]],
    se = est[["Std. Error"]],
    ci_low = est[["Estimate"]] - 1.96 * est[["Std. Error"]],
    ci_high = est[["Estimate"]] + 1.96 * est[["Std. Error"]],
    participant_intercept_sd = sqrt(as.numeric(VarCorr(fit)$participant)),
    logLik = as.numeric(logLik(fit)),
    converged = fit$optRes$convergence == 0,
    message = if (is.null(fit$optRes$message)) "" else fit$optRes$message
  )
})
human_out <- do.call(rbind, human_rows)
write.csv(human_out, args[[2]], row.names = FALSE)
print(human_out[, c("task", "n_participants", "n_trials", "beta_z", "ci_low", "ci_high")], row.names = FALSE)

if (length(args) == 4) {
  llm <- read.csv(args[[3]], stringsAsFactors = FALSE)
  llm_rows <- lapply(tasks, function(task) {
    d <- llm[llm$task == task, ]
    d$model <- factor(d$model)
    d$rd <- ordered(d$rd)
    d$x <- d$bc / 100
    fit <- clmm(rd ~ x + (1 | model), data = d, link = "logit")
    data.frame(
      task = task,
      parameter = c("beta", paste0("theta_", levels(d$rd)[-nlevels(d$rd)])),
      value = c(unname(fit$beta[["x"]]), unname(fit$Theta))
    )
  })
  write.csv(do.call(rbind, llm_rows), args[[4]], row.names = FALSE)
}
