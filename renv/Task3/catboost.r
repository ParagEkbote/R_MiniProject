# Set user library path (if needed)
.libPaths("~/R/x86_64-pc-linux-gnu-library/4.3")

# List of required packages
packages <- c("data.table", "caret", "catboost", "ggplot2", "pROC", "e1071", "reshape2")

# Install missing packages
installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

# Load all packages
invisible(lapply(packages, library, character.only = TRUE))


# Load dataset
df <- fread("transformed_land_mines.csv")

# Prepare data
X <- df[, !c("M", "V"), with=FALSE]
y <- df$M - 1  # Adjust labels

# Train-test split
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index]
X_test <- X[-train_index]
y_train <- y[train_index]
y_test <- y[-train_index]

# Prepare CatBoost pool
train_pool <- catboost.load_pool(data = X_train, label = y_train)
test_pool <- catboost.load_pool(data = X_test, label = y_test)

# Train model
params <- list(
  iterations = 250,
  eval_metric = "Accuracy",
  verbose = 50
)

start_time <- Sys.time()
model <- catboost.train(train_pool, NULL, params = params)
end_time <- Sys.time()

# Predict
y_pred <- catboost.predict(model, test_pool)
y_pred <- as.integer(y_pred)

# Evaluate
accuracy <- sum(y_pred == y_test) / length(y_test)
cat(sprintf("✅ CatBoost Accuracy: %.4f\n", accuracy))
cat(sprintf("🕒 Training Time: %.2f seconds\n", as.numeric(difftime(end_time, start_time, units = "secs"))))
print(classification_report <- confusionMatrix(as.factor(y_pred), as.factor(y_test)))

# Confusion matrix visualization
cm <- table(Predicted = y_pred, Actual = y_test)
cm_df <- as.data.frame(cm)

ggplot(data = cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1.5, color = "black") +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal()

# Save model
catboost.save_model(model, "catboost_model.cbm")
