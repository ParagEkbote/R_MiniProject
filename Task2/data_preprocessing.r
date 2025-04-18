# Dependency installation (user-preferred method)
if (!require("data.table")) install.packages("data.table", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)
if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
if (!require("reshape2")) install.packages("reshape2", dependencies = TRUE)
if (!require("Rtsne")) install.packages("Rtsne", dependencies = TRUE)
if (!require("corrplot")) install.packages("corrplot", dependencies = TRUE)
if (!require("readr")) install.packages("readr", dependencies = TRUE)

# Load libraries
library(data.table)
library(caret)
library(ggplot2)
library(dplyr)
library(e1071)
library(reshape2)
library(Rtsne)
library(corrplot)
library(readr)

# Load dataset
df <- read_csv("land_mines_dataset.csv")

# Drop "target" from features if known
X <- df %>% select(-target)  # Replace with actual target column
y <- df$target               # Replace with actual target column

# Missing values
missing_values <- colSums(is.na(X))
print("Missing Values Per Column:")
print(missing_values)

# Duplicates
duplicates <- sum(duplicated(df))
cat("Total Duplicates:", duplicates, "\n")

# Log transform 'V'
df <- df %>%
  mutate(V_log = log1p(V))

# Skewness and Kurtosis
stats_df <- data.frame(
  Feature = c("V_log", "H", "S"),
  Skewness = c(skewness(df$V_log), skewness(df$H), skewness(df$S)),
  Kurtosis = c(kurtosis(df$V_log), kurtosis(df$H), kurtosis(df$S))
)

# Save transformed data
write_csv(df, "transformed_land_mines.csv")

# Plot original vs log V
par(mfrow = c(1, 2))
hist(df$V, breaks = 20, col = "blue", main = "Original V", xlab = "V")
hist(df$V_log, breaks = 20, col = "blue", main = "Log-Transformed V", xlab = "V_log")

# Print stats
print(stats_df)
