# VISTA Human Study - Mixed-Effects Analysis
# Run from VISTA/ directory:  Rscript human_study/tools/analysis.R
#
# Required packages: install.packages(c("lme4", "lmerTest", "broom.mixed", "dplyr"))

suppressMessages({
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(dplyr)
})

# ---- Load -------------------------------------------------------------------
DATA_PATH <- "human_study/analysis_ready.csv"
OUT_DIR   <- "human_study/results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

# Apply pre-registered exclusion rule
d <- d %>% filter(excluded == 0)

cat("After exclusions:", nrow(d), "decision rows from",
    length(unique(d$participant_id)), "participants\n\n")

# Factors
d$participant_id <- factor(d$participant_id)
d$scenario_id    <- factor(d$scenario_id)
d$domain         <- factor(d$domain)
d$axis           <- factor(d$axis)

# ---- Model 1: Modifier main effect only -------------------------------------
cat("=== Model 1: choice ~ is_modified ===\n")
m1 <- glmer(choice ~ is_modified + (1 | participant_id) + (1 | scenario_id),
            data = d, family = binomial)
print(summary(m1)$coefficients)
cat("\n")

# ---- Model 2: Values only ---------------------------------------------------
# Schwartz centered scores sum to ~0 per participant (ipsatization). Drop one
# value as the implicit reference. Hedonism is the conventional drop. Document
# this choice in the paper.
cat("=== Model 2: choice ~ 9 values (hedonism is reference) ===\n")
value_terms <- paste0("centered_",
  c("self_direction","power","universalism","achievement","security",
    "stimulation","conformity","tradition","benevolence"))
f2 <- as.formula(paste("choice ~",
                       paste(value_terms, collapse = " + "),
                       "+ (1 | participant_id) + (1 | scenario_id)"))
m2 <- glmer(f2, data = d, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
print(summary(m2)$coefficients)
cat("\n")

# ---- Model 3: Values + modifier (PRIMARY confirmatory test) -----------------
cat("=== Model 3 (PRIMARY): choice ~ values + is_modified + domain ===\n")
f3 <- as.formula(paste("choice ~",
                       paste(value_terms, collapse = " + "),
                       "+ is_modified + domain",
                       "+ (1 | participant_id) + (1 | scenario_id)"))
m3 <- glmer(f3, data = d, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
m3_tab <- broom.mixed::tidy(m3, conf.int = TRUE, effects = "fixed")
m3_tab$odds_ratio    <- exp(m3_tab$estimate)
m3_tab$or_lower      <- exp(m3_tab$conf.low)
m3_tab$or_upper      <- exp(m3_tab$conf.high)
print(m3_tab)
write.csv(m3_tab, file.path(OUT_DIR, "model3_primary.csv"), row.names = FALSE)
cat("\nWrote", file.path(OUT_DIR, "model3_primary.csv"), "\n\n")

# ---- Model 4: Value x modifier interaction (exploratory) --------------------
cat("=== Model 4 (exploratory): value x is_modified interactions ===\n")
interactions <- paste(paste0(value_terms, ":is_modified"), collapse = " + ")
f4 <- as.formula(paste("choice ~",
                       paste(value_terms, collapse = " + "),
                       "+ is_modified +", interactions,
                       "+ (1 | participant_id) + (1 | scenario_id)"))
m4 <- glmer(f4, data = d, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
m4_tab <- broom.mixed::tidy(m4, conf.int = TRUE, effects = "fixed")
m4_tab$odds_ratio <- exp(m4_tab$estimate)
print(m4_tab)
write.csv(m4_tab, file.path(OUT_DIR, "model4_interactions.csv"), row.names = FALSE)
cat("\n")

# ---- Per-axis modifier effects (exploratory) --------------------------------
cat("=== Per-axis modifier effects (exploratory) ===\n")
# Use only modified items + baseline; drop other modified items per pass
per_axis_results <- list()
for (ax in setdiff(unique(d$axis), "")) {
  dd <- d %>% filter(axis == "" | axis == ax)
  dd$is_modified <- as.integer(dd$axis == ax)
  if (sum(dd$is_modified) < 20) {
    cat("  skip", ax, "(too few modified responses)\n")
    next
  }
  fA <- as.formula(paste("choice ~",
                         paste(value_terms, collapse = " + "),
                         "+ is_modified",
                         "+ (1 | participant_id) + (1 | scenario_id)"))
  mA <- glmer(fA, data = dd, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
  s <- summary(mA)$coefficients
  row <- s["is_modified", , drop = FALSE]
  per_axis_results[[ax]] <- data.frame(
    axis      = ax,
    estimate  = row[, "Estimate"],
    se        = row[, "Std. Error"],
    z         = row[, "z value"],
    p         = row[, "Pr(>|z|)"],
    odds_ratio = exp(row[, "Estimate"])
  )
}
per_axis_df <- bind_rows(per_axis_results)
per_axis_df$p_bh <- p.adjust(per_axis_df$p, method = "BH")
print(per_axis_df)
write.csv(per_axis_df, file.path(OUT_DIR, "per_axis_modifier.csv"), row.names = FALSE)
cat("\nWrote", file.path(OUT_DIR, "per_axis_modifier.csv"), "\n\n")

# ---- Model comparison (likelihood ratio test) -------------------------------
cat("=== Likelihood ratio: does adding 'is_modified' improve fit? ===\n")
print(anova(m2, m3))
cat("\nAll results saved in", OUT_DIR, "\n")
