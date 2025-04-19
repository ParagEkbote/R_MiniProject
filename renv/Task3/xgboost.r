# Load necessary libraries
library(data.table)
library(xgboost)
library(caret)
library(ggplot2)
library(e1071)
library(pROC)

# Load dataset
df <- fread("transformed_land_mines.csv")

# Prepare data
X <- df[, !c("M", "V"), with = FALSE]
y <- df$M - 1  # Adjusting labels if needed

# Train-test split (80-20)
set.seed(42)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- as.matrix(X[train_idx])
X_test <- as.matrix(X[-train_idx])
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Define model parameters
params <- list(
  booster = "dart",
  objective = "multi:softmax",
  num_class = length(unique(y)),
  eval_metric = "mlogloss",
  max_depth = 10,
  eta = 0.7,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 18,
  alpha = 0.6
)

# Train model
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  verbose = 0
)

# Predictions
y_pred <- predict(model, X_test)

# Accuracy
accuracy <- mean(y_pred == y_test)
cat(sprintf("✅ XGBoost Accuracy: %.4f\n", accuracy))

# Classification report
conf_mat <- confusionMatrix(as.factor(y_pred), as.factor(y_test))
print(conf_mat)

# Confusion Matrix heatmap
cm_df <- as.data.frame(conf_mat$table)
ggplot(data = cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal()

# Save model
xgb.save(model, "xgboost_model.model")
