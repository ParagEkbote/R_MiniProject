# Load libraries
library(data.table)
library(caret)
library(ggplot2)
library(randomForest)  # Add this line to load the randomForest package

# Load dataset
df <- fread("transformed_land_mines.csv")

# Prepare data
X <- df[, !c("M", "V"), with = FALSE]
y <- df$M - 1  # Adjusting target

# Train-test split (with stratification)
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index,]  # Add comma for proper subsetting
X_test <- X[-train_index,]  # Add comma for proper subsetting
y_train <- y[train_index]
y_test <- y[-train_index]

# Convert to data.frame for training
train_data <- data.frame(X_train, y = as.factor(y_train))
test_data <- data.frame(X_test)

# Train the model
final_model <- randomForest(
  y ~ .,
  data = train_data,
  ntree = 100,
  maxnodes = 2^15,  # Mimic max_depth = 15
  nodesize = 1,     # min_samples_leaf = 1
  mtry = floor(sqrt(ncol(X_train))),  # max_features = "sqrt"
  importance = TRUE
)

# Predict
y_pred <- predict(final_model, newdata = test_data)
y_pred <- as.numeric(as.character(y_pred))  # Ensure labels match type

# Evaluate
accuracy <- mean(y_pred == y_test)
cat(sprintf("✅ RandomForest Accuracy: %.4f\n", accuracy))
cat("📊 Classification Report:\n")
print(confusionMatrix(as.factor(y_pred), as.factor(y_test)))

# Confusion matrix heatmap
cm <- table(Actual = y_test, Predicted = y_pred)
cm_df <- as.data.frame(cm)

ggplot(data = cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1.5, color = "black") +
  scale_fill_gradient(low = "peachpuff", high = "darkorange") +
  labs(title = "Confusion Matrix - Random Forest", x = "Predicted", y = "Actual") +
  theme_minimal()

# Save model
saveRDS(final_model, "randomforest_model.rds")
cat("🎉 Model saved as 'randomforest_model.rds'\n")
