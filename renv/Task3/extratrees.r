# Load necessary libraries
library(randomForest)
library(caret)
library(ggplot2)
library(e1071)

# Load dataset
df <- read.csv("transformed_land_mines.csv")

# Prepare data
X <- df[, !(names(df) %in% c("M", "V"))]  # Drop columns "M" and "V"
y <- df$M - 1  # Adjust the target variable if necessary (subtract 1 from M)

# Split into train-test sets (80% train, 20% test)
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Initialize and train the Extra Trees model (randomForest as alternative)
start_time <- Sys.time()
model <- randomForest(x = X_train, y = as.factor(y_train), ntree = 250, randomState = 42)
end_time <- Sys.time()

# Evaluate the model
y_pred <- predict(model, X_test)
accuracy <- sum(y_pred == y_test) / length(y_test)
print(paste("✅ ExtraTrees Accuracy: ", round(accuracy, 4)))

# Print classification report (using confusion matrix for now)
conf_matrix <- table(Predicted = y_pred, Actual = y_test)
print(conf_matrix)

# Confusion matrix visualization
library(ggplot2)
conf_matrix_df <- as.data.frame(as.table(conf_matrix))
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Extra Trees", x = "Predicted", y = "Actual")

# Save model (serialized)
saveRDS(model, "extratrees_model.rds")
print("🎉 Model saved as 'extratrees_model.rds'")

# Training Time
print(paste("🕒 Training Time: ", as.numeric(difftime(end_time, start_time, units = "secs")), " seconds"))
