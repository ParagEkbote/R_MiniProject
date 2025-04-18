# Load libraries
library(data.table)
library(caret)
library(ggplot2)
library(e1071)
library(reshape2)

# Load dataset
df <- fread("transformed_land_mines.csv")

# Prepare data
X <- df[, !c("M", "V_log"), with = FALSE]
y <- df$M - 1  # Adjust labels

# Train-test split (with stratification)
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index]
X_test <- X[-train_index]
y_train <- y[train_index]
y_test <- y[-train_index]

# Combine train data for glm
train_data <- data.frame(X_train, y = as.factor(y_train))
test_data <- data.frame(X_test)

# Train logistic regression model
start_time <- Sys.time()
model <- glm(y ~ ., data = train_data, family = binomial)
end_time <- Sys.time()

# Predict (as class labels)
prob_pred <- predict(model, test_data, type = "response")
y_pred <- ifelse(prob_pred > 0.5, 1, 0)

# Evaluate
accuracy <- mean(y_pred == y_test)
cat(sprintf("✅ Logistic Regression Accuracy: %.4f\n", accuracy))
cat(sprintf("🕒 Training Time: %.2f seconds\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

# Classification report (confusion matrix and metrics)
conf_mat <- confusionMatrix(as.factor(y_pred), as.factor(y_test))
print(conf_mat)

# Confusion matrix heatmap
cm_df <- as.data.frame(conf_mat$table)
ggplot(data = cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1.5, color = "black") +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
  labs(title = "Confusion Matrix - Logistic Regression", x = "Predicted", y = "Actual") +
  theme_minimal()

# Save model
saveRDS(model, "logistic_regression_model.rds")
cat("🎉 Model saved as 'logistic_regression_model.rds'\n")
