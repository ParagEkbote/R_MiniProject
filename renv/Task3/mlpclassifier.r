# Install required packages if not installed
if (!require("nnet")) install.packages("nnet", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)

# Load libraries
library(nnet)
library(caret)
library(e1071)
library(ggplot2)

# Load dataset
df <- read.csv("transformed_land_mines.csv")

# Prepare data
X <- df[, !(names(df) %in% c("M", "V"))]  # Drop "M" and "V"
y <- df$M - 1                            # Shift labels from 1–5 to 0–4
y <- as.factor(y)                        # Convert target to factor

# Standardize features
preProc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preProc, X)

# Combine for caret splitting
data <- cbind(X_scaled, y)

# Train-test split
set.seed(42)
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train MLP model (1 hidden layer due to `nnet` limitation)
start_time <- Sys.time()
mlp_model <- nnet(
  y ~ ., 
  data = trainData,
  size = 16,                # Approximation of hidden layers; nnet supports 1 hidden layer
  decay = 1.5e-5,           # Weight decay (alpha equivalent)
  maxit = 587,              # Number of iterations
  trace = FALSE             # Suppress output
)
end_time <- Sys.time()

# Predict
preds <- predict(mlp_model, testData[, -ncol(testData)], type = "class")

# Accuracy
acc <- sum(preds == testData$y) / nrow(testData)
cat(sprintf("✅ Final MLP Accuracy: %.4f\n", acc))
cat(sprintf("🕒 Training Time: %.2f seconds\n\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

# Classification report
conf_matrix <- confusionMatrix(preds, testData$y)
print(conf_matrix)

# Confusion matrix plot
conf_df <- as.data.frame(conf_matrix$table)
colnames(conf_df) <- c("Predicted", "Actual", "Freq")

ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - MLPClassifier (R)", x = "Actual", y = "Predicted") +
  theme_minimal()

# Save model and scaler
saveRDS(mlp_model, "mlp_model_nnet.rds")
saveRDS(preProc, "scaler_nnet.rds")
cat("🎉 Model and scaler saved as RDS files.\n")
