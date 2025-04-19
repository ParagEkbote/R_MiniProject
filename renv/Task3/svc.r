# Install required packages if needed
if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)

# Load libraries
library(e1071)
library(caret)
library(ggplot2)

# Load dataset
df <- read.csv("transformed_land_mines.csv")

# Prepare data
X <- df[, !(names(df) %in% c("M", "V"))]
y <- as.factor(df$M - 1)

# Combine for caret processing
data <- cbind(X, y)

# Train-test split
set.seed(42)
train_index <- createDataPartition(data$y, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the SVM model with RBF kernel
start_time <- Sys.time()
svm_model <- svm(
  y ~ .,
  data = train_data,
  kernel = "radial",
  probability = TRUE
)
end_time <- Sys.time()

# Predict on test data
preds <- predict(svm_model, newdata = test_data[, -ncol(test_data)])

# Accuracy
acc <- sum(preds == test_data$y) / nrow(test_data)
cat(sprintf("✅ SVM Accuracy: %.4f\n", acc))
cat(sprintf("🕒 Training Time: %.2f seconds\n\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

# Classification report (confusion matrix)
conf_matrix <- confusionMatrix(preds, test_data$y)
print(conf_matrix)

# Confusion matrix heatmap
cm_df <- as.data.frame(conf_matrix$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")

ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "#F0E6F6", high = "#5E3C99") +
  labs(title = "Confusion Matrix - SVM (RBF Kernel)", x = "Actual", y = "Predicted") +
  theme_minimal()

# Save model
saveRDS(svm_model, "svm_model_rbf.rds")
cat("🎉 Model saved as 'svm_model_rbf.rds'\n")
