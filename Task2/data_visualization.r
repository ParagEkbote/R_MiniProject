# Set a writable library path
.libPaths("~/R/x86_64-pc-linux-gnu-library/4.3")
print(.libPaths())  # Optional: confirm the path

# Dependency installation (user-preferred method)
if (!require("data.table")) install.packages("data.table", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)
if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
if (!require("reshape2")) install.packages("reshape2", dependencies = TRUE)
if (!require("Rtsne")) install.packages("Rtsne", dependencies = TRUE)
if (!require("corrplot")) install.packages("corrplot", dependencies = TRUE)

# Load libraries after installation
library(data.table)
library(caret)
library(ggplot2)
library(dplyr)
library(e1071)
library(reshape2)
library(Rtsne)
library(corrplot)

# Load dataset
file_path <- "transformed_land_mines.csv"
df <- fread(file_path, data.table = FALSE)

# Drop target and V
X <- df %>% select(-M, -V)

# Histograms
X_long <- melt(X)
ggplot(X_long, aes(x = value)) +
  facet_wrap(~variable, scales = "free") +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  theme_minimal()
ggsave("histograms_excluding_V.png")

# Boxplots
X_long <- melt(X)
ggplot(X_long, aes(x = variable, y = value)) +
  geom_boxplot(fill = "skyblue", outlier.color = "red") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))
ggsave("boxplot_excluding_V.png")

# Correlation heatmap
cor_matrix <- cor(X, use = "complete.obs")
png("correlation_heatmap_excluding_V.png", width = 800, height = 600)
corrplot(cor_matrix, method = "color", type = "lower", tl.cex = 0.8, number.cex = 0.7)
dev.off()

cat("Visualizations saved (excluding V).\n")

# PCA
X_scaled <- scale(X)
pca <- prcomp(X_scaled, center = TRUE, scale. = TRUE)
pca_df <- as.data.frame(pca$x[, 1:2])
pca_df$label <- as.factor(df$M - 1)

# t-SNE
set.seed(42)
tsne <- Rtsne(X_scaled, dims = 2, perplexity = 30, verbose = TRUE, max_iter = 1000)
tsne_df <- as.data.frame(tsne$Y)
tsne_df$label <- as.factor(df$M - 1)

# Plot PCA and t-SNE
p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = label)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "PCA Projection")

p2 <- ggplot(tsne_df, aes(x = V1, y = V2, color = label)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "t-SNE Projection")

library(gridExtra)
png("pca_tsne_projection.png", width = 1200, height = 600)
grid.arrange(p1, p2, ncol = 2)
dev.off()