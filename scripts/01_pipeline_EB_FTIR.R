# ==========================================================
# Machine-learning pipeline for predicting BE (EB) from FTIR
# Models: Random Forest (ranger/caret) + Decision Tree (rpart)
# Outputs: Train/Test metrics, 10-fold CV, bootstrap (B=500),
#          2D heatmaps (top-2 predictors), tree plots,
#          Bland–Altman (bias, LoA95, R2, RMSE, MAE, r, p).
# ==========================================================

suppressPackageStartupMessages({
  library(dplyr);  library(tibble);  library(tidyr)
  library(ggplot2); library(viridisLite); library(readr)
  library(caret);   library(ranger)
  library(rpart);   library(rpart.plot)
  library(scales)
})

set.seed(123)

# -----------------------------
# 0) Utility functions
# -----------------------------
R2 <- function(y, yhat){
  sy <- suppressWarnings(sd(y))
  if (!is.finite(sy) || sy == 0) return(NA_real_)
  suppressWarnings(cor(y, yhat)^2)
}
RMSE <- function(y,yhat) sqrt(mean((y - yhat)^2))
MAE  <- function(y,yhat) mean(abs(y - yhat))
safe_min_node <- function(bt) if ("min.node.size" %in% names(bt)) bt$min.node.size else 5

# Tuning grid (4 FTIR predictors)
tg_grid <- expand.grid(
  mtry         = 1:4,
  splitrule    = "variance",
  min.node.size = c(2,3,4,5)
)

# Heatmap for top-2 variables (others set to 0)
heatmap_from_stability <- function(df, top2_vars, ngrid = 180){
  stopifnot(length(top2_vars) == 2)
  set.seed(123)
  rf_all <- caret::train(
    EB ~ ., data = df, method = "ranger",
    trControl = caret::trainControl(method = "cv", number = 10),
    tuneGrid  = tg_grid, importance = "permutation",
    num.trees = 500, metric = "RMSE"
  )
  best <- rf_all$bestTune
  fit  <- ranger::ranger(
    EB ~ ., data = df,
    mtry        = best$mtry,
    splitrule   = as.character(best$splitrule),
    min.node.size = safe_min_node(best),
    num.trees   = 500, seed = 999
  )

  v1   <- top2_vars[1]; v2 <- top2_vars[2]
  rest <- setdiff(names(df)[-1], c(v1, v2))

  grid <- expand.grid(
    x1 = seq(min(df[[v1]]), max(df[[v1]]), length.out = ngrid),
    x2 = seq(min(df[[v2]]), max(df[[v2]]), length.out = ngrid)
  )
  names(grid) <- c(v1, v2)

  add0  <- as.data.frame(matrix(0, nrow = nrow(grid), ncol = length(rest)))
  names(add0) <- rest
  newdat <- cbind(grid, add0)
  newdat$EB_pred <- predict(fit, data = newdat)$predictions

  ggplot(newdat, aes(x = .data[[v1]], y = .data[[v2]], fill = EB_pred)) +
    geom_tile() +
    stat_contour(aes(z = EB_pred),
                 color = "white", linewidth = 0.35,
                 bins = 12, show.legend = FALSE) +
    scale_fill_viridis_c(option = "inferno", direction = -1,
                         name = "EB\npred.") +
    labs(x = v1, y = v2) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_blank(),
          plot.subtitle = element_blank())
}

# Bootstrap stability (B = 500)
stab_importance <- function(df, B = 500, num.trees = 500, seed = 2024){
  set.seed(seed)
  cv_all <- caret::train(
    EB ~ ., data = df, method = "ranger",
    trControl = caret::trainControl(method = "cv", number = 10),
    tuneGrid  = tg_grid, importance = "permutation",
    num.trees = num.trees, metric = "RMSE"
  )
  bt   <- cv_all$bestTune
  vars <- setdiff(names(df), "EB")
  IMP  <- matrix(NA_real_, nrow = B, ncol = length(vars),
                 dimnames = list(NULL, vars))
  RANK <- matrix(NA_integer_, nrow = B, ncol = length(vars),
                 dimnames = list(NULL, vars))

  for (b in 1:B){
    idx <- sample(seq_len(nrow(df)), replace = TRUE)
    fit_b <- ranger::ranger(
      EB ~ ., data = df[idx, ],
      mtry = bt$mtry, splitrule = as.character(bt$splitrule),
      min.node.size = safe_min_node(bt),
      num.trees = num.trees, importance = "permutation",
      seed = seed + b
    )
    impb <- ranger::importance(fit_b)
    IMP[b, names(impb)] <- impb
    RANK[b, ] <- rank(-IMP[b, ], ties.method = "min")
  }

  as_tibble(IMP) |>
    pivot_longer(everything(), names_to = "Variable", values_to = "Imp") |>
    group_by(Variable) |>
    summarise(Mean = mean(Imp, na.rm = TRUE),
              SD   = sd(Imp,   na.rm = TRUE), .groups = "drop") |>
    left_join(
      as_tibble(RANK) |>
        pivot_longer(everything(), names_to = "Variable", values_to = "rk") |>
        group_by(Variable) |>
        summarise(Top1_pct = mean(rk == 1,    na.rm = TRUE)*100,
                  Top2_pct = mean(rk <= 2,    na.rm = TRUE)*100,
                  .groups = "drop"),
      by = "Variable"
    ) |>
    mutate(CI95_low  = Mean - 1.96*SD,
           CI95_high = Mean + 1.96*SD) |>
    arrange(desc(Top2_pct), desc(Mean))
}

# Top-2 variables for heatmap (forces C1420 as X if present)
pick_top2_vars <- function(stab_tbl){
  stab_tbl <- stab_tbl |> arrange(desc(Top2_pct), desc(Mean))
  top2 <- stab_tbl$Variable[1:2]
  if ("C1420" %in% top2){
    top2 <- c("C1420", setdiff(top2, "C1420"))
  }
  top2
}

# Random Forest: 70/30 split + global 10-fold CV
train_rf_full <- function(df, name_prefix){
  set.seed(123)
  split_idx <- caret::createDataPartition(df$EB, p = 0.7, list = FALSE)
  tr <- df[ split_idx, ]
  te <- df[-split_idx, ]

  ctrl  <- caret::trainControl(method = "cv", number = 10)
  rf_cv <- caret::train(
    EB ~ ., data = tr, method = "ranger",
    trControl  = ctrl,
    tuneGrid   = tg_grid,
    importance = "permutation",
    num.trees  = 500, metric = "RMSE"
  )
  best <- rf_cv$bestTune

  set.seed(123)
  rf_fit <- ranger::ranger(
    EB ~ ., data = tr,
    mtry = best$mtry,
    splitrule    = as.character(best$splitrule),
    min.node.size = safe_min_node(best),
    num.trees    = 500, importance = "permutation", seed = 123
  )

  pred_tr <- predict(rf_fit, data = tr)$predictions
  pred_te <- predict(rf_fit, data = te)$predictions

  metrics_split <- bind_rows(
    tibble(Set = "Train",
           R2   = R2(tr$EB, pred_tr),
           RMSE = RMSE(tr$EB, pred_tr),
           MAE  = MAE(tr$EB, pred_tr)),
    tibble(Set = "Test",
           R2   = R2(te$EB, pred_te),
           RMSE = RMSE(te$EB, pred_te),
           MAE  = MAE(te$EB, pred_te))
  )
  write_csv(metrics_split, paste0(name_prefix, "_metrics_split.csv"))

  set.seed(123)
  rf_cv_all <- caret::train(
    EB ~ ., data = df, method = "ranger",
    trControl = caret::trainControl(method = "cv", number = 10),
    tuneGrid  = tg_grid, importance = "permutation",
    num.trees = 500, metric = "RMSE"
  )
  best_row <- rf_cv_all$results |>
    filter(mtry        == rf_cv_all$bestTune$mtry,
           splitrule   == rf_cv_all$bestTune$splitrule,
           min.node.size == rf_cv_all$bestTune$min.node.size) |>
    arrange(RMSE) |>
    slice(1)

  metrics_cv10 <- tibble(
    R2   = best_row$Rsquared,
    RMSE = best_row$RMSE,
    MAE  = best_row$MAE
  )
  write_csv(metrics_cv10, paste0(name_prefix, "_metrics_cv10.csv"))

  list(model = rf_fit, best = best, split_idx = split_idx,
       metrics_split = metrics_split, metrics_cv10 = metrics_cv10)
}

# Decision Tree: large tree + pruning to ~8 leaves, nested CV with OOF
fit_pruned_tree <- function(df, target = "EB", leaves_target = 8L){
  set.seed(123)
  idx_te <- createDataPartition(df[[target]], p = 0.30, list = FALSE)
  te <- df[idx_te, , drop = FALSE]
  tr <- df[-idx_te, , drop = FALSE]

  ctrl_big <- rpart.control(cp = 1e-4, minsplit = 3, minbucket = 2,
                            maxdepth = 30, xval = 0)
  fit_big  <- rpart(stats::as.formula(paste(target, "~ .")),
                    data = tr, method = "anova", control = ctrl_big)

  tab <- fit_big$cptable
  if (is.null(tab) || nrow(tab) == 0){
    fit_8 <- fit_big
  } else {
    i <- which(tab[, "nsplit"] == (leaves_target - 1L))
    if (!length(i))
      i <- which.min(abs(tab[, "nsplit"] - (leaves_target - 1L)))
    fit_8 <- prune(fit_big, cp = tab[i, "CP"])
  }

  pred_tr <- predict(fit_8, newdata = tr)
  pred_te <- predict(fit_8, newdata = te)
  metrics_split <- bind_rows(
    tibble(Set = "Train",
           R2   = R2(tr[[target]], pred_tr),
           RMSE = RMSE(tr[[target]], pred_tr),
           MAE  = MAE(tr[[target]], pred_tr)),
    tibble(Set = "Test",
           R2   = R2(te[[target]], pred_te),
           RMSE = RMSE(te[[target]], pred_te),
           MAE  = MAE(te[[target]], pred_te))
  )

  set.seed(123)
  folds <- createFolds(df[[target]], k = 10, returnTrain = FALSE)
  oof   <- rep(NA_real_, nrow(df)); fold_rows <- list()

  for (k in seq_along(folds)){
    idx_val <- folds[[k]]
    dtr <- df[-idx_val, , drop = FALSE]
    dva <- df[ idx_val, , drop = FALSE]

    fb  <- rpart(stats::as.formula(paste(target, "~ .")),
                 data = dtr, method = "anova", control = ctrl_big)
    cpt <- fb$cptable
    if (is.null(cpt) || nrow(cpt) == 0){
      f8 <- fb
    } else {
      ii <- which(cpt[, "nsplit"] == (leaves_target - 1L))
      if (!length(ii))
        ii <- which.min(abs(cpt[, "nsplit"] - (leaves_target - 1L)))
      f8 <- prune(fb, cp = cpt[ii, "CP"])
    }
    p_tr <- predict(f8, newdata = dtr)
    p_va <- predict(f8, newdata = dva)
    oof[idx_val] <- p_va

    fold_rows[[k]] <- tibble(
      fold       = k,
      leaves     = sum(f8$frame$var == "<leaf>"),
      R2_train   = R2(dtr[[target]], p_tr),
      RMSE_train = RMSE(dtr[[target]], p_tr),
      MAE_train  = MAE(dtr[[target]], p_tr),
      R2_val     = R2(dva[[target]], p_va),
      RMSE_val   = RMSE(dva[[target]], p_va),
      MAE_val    = MAE(dva[[target]], p_va)
    )
  }

  cv10_folds   <- bind_rows(fold_rows)
  cv10_summary <- cv10_folds %>%
    summarise(
      folds        = n(),
      leaves_mean  = mean(leaves),
      leaves_sd    = sd(leaves),
      R2_val_mean  = mean(R2_val,   na.rm = TRUE),
      RMSE_val_mean= mean(RMSE_val, na.rm = TRUE),
      MAE_val_mean = mean(MAE_val,  na.rm = TRUE)
    )
  oof_tbl <- tibble(Obs = df[[target]], Pred_OOF = oof)
  oof_metrics <- tibble(
    R2_OOF   = R2(oof_tbl$Obs,  oof_tbl$Pred_OOF),
    RMSE_OOF = RMSE(oof_tbl$Obs, oof_tbl$Pred_OOF),
    MAE_OOF  = MAE(oof_tbl$Obs,  oof_tbl$Pred_OOF)
  )

  list(fit = fit_8, tr = tr, te = te,
       metrics_split = metrics_split,
       cv10_folds = cv10_folds, cv10_summary = cv10_summary,
       oof_tbl = oof_tbl, oof_metrics = oof_metrics)
}

# Tree plot with inferno palette
plot_tree_inferno <- function(fit, file_png, title = "Decision Tree", dpi = 200){
  pal   <- viridisLite::inferno(100, direction = -1)
  yvals <- fit$frame$yval
  idx   <- round(scales::rescale(yvals, to = c(1, 100)))
  idx[idx < 1]   <- 1
  idx[idx > 100] <- 100
  box_cols <- pal[idx]

  rgbm <- t(col2rgb(box_cols))/255
  lum  <- 0.2126*rgbm[,1] + 0.7152*rgbm[,2] + 0.0722*rgbm[,3]
  text_cols <- ifelse(lum < 0.45, "white", "black")

  png(file_png, width = 1800, height = 1200, res = dpi)
  rpart.plot::rpart.plot(
    fit,
    type = 5, extra = 101, under = TRUE, fallen.leaves = TRUE,
    branch.lty = 1, shadow.col = rgb(0,0,0,0.15),
    box.col = box_cols, border.col = "grey15",
    col = text_cols, roundint = FALSE, tweak = 1.05,
    main = title
  )
  dev.off()
}

save_rules_varimp <- function(fit, prefix){
  if (!is.null(fit$variable.importance)){
    imp <- tibble(
      Variable   = names(fit$variable.importance),
      Importance = as.numeric(fit$variable.importance)
    ) |>
      arrange(desc(Importance))
    write_csv(imp, paste0(prefix, "_varimp.csv"))
  }
  leaves <- row.names(fit$frame)[fit$frame$var == "<leaf>"]
  rules <- lapply(leaves, function(n){
    p <- path.rpart(fit, nodes = as.numeric(n), pretty = 0)[[1]]
    paste(p, collapse = " & ")
  })
  rules_tbl <- tibble(
    leaf = leaves,
    rule = unlist(rules),
    yval = fit$frame$yval[fit$frame$var == "<leaf>"]
  )
  write_csv(rules_tbl, paste0(prefix, "_rules.csv"))
}

# Bland–Altman + Pearson
bland_altman <- function(obs, pred, title, file_png){
  df <- tibble(
    mean = (obs + pred)/2,
    diff = pred - obs
  )
  bias   <- mean(df$diff)
  sd_d   <- sd(df$diff)
  loa_low  <- bias - 1.96*sd_d
  loa_high <- bias + 1.96*sd_d

  ct        <- cor.test(obs, pred)
  r_pearson <- as.numeric(ct$estimate)
  p_pearson <- ct$p.value

  p <- ggplot(df, aes(x = mean, y = diff)) +
    geom_hline(yintercept = 0, linetype = 2, color = "grey50") +
    geom_point(alpha = 0.9) +
    geom_hline(yintercept = bias,    color = "black") +
    geom_hline(yintercept = loa_low, color = "red") +
    geom_hline(yintercept = loa_high,color = "red") +
    labs(title = title,
         x = "Mean (obs, pred)",
         y = "Difference (pred - obs)") +
    theme_minimal(base_size = 12)
  ggsave(file_png, p, width = 7, height = 5, dpi = 300)

  tibble(
    Model     = title,
    N         = length(obs),
    Bias      = bias,
    LoA_low   = loa_low,
    LoA_high  = loa_high,
    R2        = R2(obs, pred),
    RMSE      = RMSE(obs, pred),
    MAE       = MAE(obs, pred),
    r_pearson = r_pearson,
    p_pearson = p_pearson
  )
}

# -----------------------------
# 1) DATASETS (user-provided)
# -----------------------------
# The following CSV files are assumed to exist in "data/":
# - EB_FTIR_initial.csv:    EB, C890, C1420, C1510, C1740
# - EB_FTIR_consumption.csv: EB, C890, C1420, C1510, C1740
# - EB_FTIR_newdata.csv:    C890, C1420, C1510, C1740, EB_obs

df_ini  <- read_csv("data/EB_FTIR_initial.csv")
df_cons <- read_csv("data/EB_FTIR_consumption.csv")
new_tbl <- read_csv("data/EB_FTIR_newdata.csv")

# -----------------------------
# 2) Random Forests + Bootstrap + Heatmaps
# -----------------------------
rf_ini_res  <- train_rf_full(df_ini,  "RF_EB_INICIAL_FTIR")
rf_cons_res <- train_rf_full(df_cons, "RF_EB_CONSUMO_FTIR")

stab_ini  <- stab_importance(df_ini,  B = 500)
write_csv(stab_ini,  "RF_EB_INICIAL_FTIR_importance_bootstrap.csv")
stab_cons <- stab_importance(df_cons, B = 500)
write_csv(stab_cons, "RF_EB_CONSUMO_FTIR_importance_bootstrap.csv")

top2_ini  <- pick_top2_vars(stab_ini)
top2_cons <- pick_top2_vars(stab_cons)

p_hm_ini  <- heatmap_from_stability(df_ini,  top2_ini)
p_hm_cons <- heatmap_from_stability(df_cons, top2_cons)

ggsave("RF_EB_INICIAL_FTIR_heatmap_top2.png",
       p_hm_ini,  width = 7.5, height = 6, dpi = 600)
ggsave("RF_EB_CONSUMO_FTIR_heatmap_top2.png",
       p_hm_cons, width = 7.5, height = 6, dpi = 600)

# -----------------------------
# 3) Decision Trees + PNG
# -----------------------------
res_ini <- fit_pruned_tree(df_ini,  target = "EB", leaves_target = 8)
res_con <- fit_pruned_tree(df_cons, target = "EB", leaves_target = 8)

write_csv(res_ini$metrics_split, "DT_EB_INICIAL_FTIR_metrics_split.csv")
write_csv(res_ini$cv10_folds,    "DT_EB_INICIAL_FTIR_cv10_folds.csv")
write_csv(res_ini$cv10_summary,  "DT_EB_INICIAL_FTIR_cv10_summary.csv")
write_csv(res_ini$oof_tbl,       "DT_EB_INICIAL_FTIR_oof_predictions.csv")
write_csv(res_ini$oof_metrics,   "DT_EB_INICIAL_FTIR_oof_metrics.csv")
plot_tree_inferno(res_ini$fit,
                  "DT_EB_INICIAL_FTIR_8leaves.png",
                  "Decision Tree — EB (Initial, 8 leaves)")
save_rules_varimp(res_ini$fit, "DT_EB_INICIAL_FTIR")

write_csv(res_con$metrics_split, "DT_EB_CONSUMO_FTIR_metrics_split.csv")
write_csv(res_con$cv10_folds,    "DT_EB_CONSUMO_FTIR_cv10_folds.csv")
write_csv(res_con$cv10_summary,  "DT_EB_CONSUMO_FTIR_cv10_summary.csv")
write_csv(res_con$oof_tbl,       "DT_EB_CONSUMO_FTIR_oof_predictions.csv")
write_csv(res_con$oof_metrics,   "DT_EB_CONSUMO_FTIR_oof_metrics.csv")
plot_tree_inferno(res_con$fit,
                  "DT_EB_CONSUMO_FTIR_8leaves.png",
                  "Decision Tree — EB (Consumption, 8 leaves)")
save_rules_varimp(res_con$fit, "DT_EB_CONSUMO_FTIR")

# -----------------------------
# 4) Predictions + Bland–Altman
# -----------------------------
pred_tbl <- new_tbl %>%
  mutate(
    RF_ini  = as.numeric(predict(rf_ini_res$model,  data   = new_tbl)$predictions),
    RF_cons = as.numeric(predict(rf_cons_res$model, data   = new_tbl)$predictions),
    DT_ini  = as.numeric(predict(res_ini$fit,  newdata = new_tbl)),
    DT_cons = as.numeric(predict(res_con$fit, newdata = new_tbl))
  )
write_csv(pred_tbl, "Predicciones_EB_nuevos_datos_FTIR.csv")

ba_summary <- bind_rows(
  bland_altman(new_tbl$EB_obs, pred_tbl$RF_ini,
               "RF — Initial (FTIR)",  "BA_RF_inicial_FTIR.png"),
  bland_altman(new_tbl$EB_obs, pred_tbl$RF_cons,
               "RF — Consumption (FTIR)", "BA_RF_consumo_FTIR.png"),
  bland_altman(new_tbl$EB_obs, pred_tbl$DT_ini,
               "DT — Initial (FTIR)",  "BA_DT_inicial_FTIR.png"),
  bland_altman(new_tbl$EB_obs, pred_tbl$DT_cons,
               "DT — Consumption (FTIR)", "BA_DT_consumo_FTIR.png")
)
write_csv(ba_summary, "BlandAltman_summary_nuevos_datos_FTIR.csv")
print(ba_summary)

# -----------------------------
# 5) Log
# -----------------------------
cat("\n== Generated files ==\n",
    "- RF_EB_INICIAL_FTIR_metrics_split.csv / _metrics_cv10.csv\n",
    "- RF_EB_CONSUMO_FTIR_metrics_split.csv / _metrics_cv10.csv\n",
    "- RF_EB_INICIAL_FTIR_importance_bootstrap.csv | RF_EB_CONSUMO_FTIR_importance_bootstrap.csv\n",
    "- RF_EB_INICIAL_FTIR_heatmap_top2.png | RF_EB_CONSUMO_FTIR_heatmap_top2.png\n",
    "- DT_EB_INICIAL_FTIR_* (rules/varimp/cv10/oof/metrics, png)\n",
    "- DT_EB_CONSUMO_FTIR_* (rules/varimp/cv10/oof/metrics, png)\n",
    "- Predicciones_EB_nuevos_datos_FTIR.csv\n",
    "- BA_*_FTIR.png | BlandAltman_summary_nuevos_datos_FTIR.csv\n")

