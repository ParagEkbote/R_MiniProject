# Load libraries
library(data.table)
library(caret)
library(ggplot2)
library(catboost)

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

# Train model with tuned hyperparameters
params <- list(
  iterations = 517,
  depth = 4,
  learning_rate = 0.013302095484120483,
  l2_leaf_reg = 0.04450546192783469,
  border_count = 107,
  bagging_temperature = 0.564828924702632,
  loss_function = "MultiClass",
  eval_metric = "Accuracy",
  verbose = 50
)

start_time <- Sys.time()
model <- catboost.train(train_pool, NULL, params = params)
end_time <- Sys.time()

# Predict
y_pred <- catboost.predict(model, test_pool, prediction_type = "Class")
y_pred <- as.integer(y_pred)

# Ensure matching lengths
cat(sprintf("Length of predictions: %d | Length of test labels: %d\n", length(y_pred), length(y_test)))

# Evaluate
accuracy <- sum(y_pred == y_test) / length(y_test)
cat(sprintf("✅ Final CatBoost Accuracy: %.4f\n", accuracy))
cat(sprintf("🕒 Training Time: %.2f seconds\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

# Ensure matching factor levels
y_pred <- factor(y_pred, levels = sort(unique(y_test)))
y_test <- factor(y_test, levels = sort(unique(y_test)))

# Confusion matrix
print(classification_report <- confusionMatrix(y_pred, y_test))

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
