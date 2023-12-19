# Install necessary packages
install.packages("dendextend")  # Install dendextend package for better visualization of dendrogram
install.packages("ggplot2")     # Install ggplot2 for plotting
install.packages("dplyr")       # Install dplyr for data manipulation
install.packages("factoextra")  # Install factoextra for visualizing hierarchical clustering results
install.packages("proxy")       # Install proxy for calculating dissimilarity matrix
install.packages("cluster")     # Install cluster for hierarchical clustering algorithms
install.packages("glmnet")      # Install glmnet for Lasso regression
install.packages("ggplot2")     # Install ggplot2 for additional plotting

# Load necessary libraries
library(dendextend)
library(ggplot2)
library(dplyr)
library(factoextra)
library(proxy)
library(cluster)
library(glmnet)

# Read the dataset
mydata <- read.csv("C:/Users/Pavan/Desktop/AI/Mall_Customers.csv")

# Display the structure and summary of the dataset
str(mydata)
summary(mydata)

# Check and handle missing values
missing_values <- colSums(is.na(mydata))
print("Missing Values:")
print(missing_values)
mydata <- na.omit(mydata)
mydata <- dplyr::distinct(mydata)

# Select numeric columns for hierarchical clustering
numeric_columns <- dplyr::select(mydata, where(is.numeric))

# Calculate the dissimilarity matrix and perform hierarchical clustering
dist_matrix <- proxy::dist(as.matrix(numeric_columns), method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

# Plot histograms of selected numeric variables
selected_variables <- c('Age', 'Annual.Income..k..', 'Spending.Score..1.100.')
selected_data <- mydata %>%
  select(all_of(selected_variables))
data_melted <- reshape2::melt(selected_data)
ggplot(data_melted, aes(x = value)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Histograms of Selected Variables", x = "Value", y = "Frequency")

# Visualize the hierarchical clustering dendrogram
dend <- as.dendrogram(hc)
dend <- dendextend::color_branches(dend, k = 4)
plot(dend)

# Visualize hierarchical clustering results using factor variable
clusters <- cutree(hc, k = 4)
factoextra::fviz_cluster(list(data = as.matrix(numeric_columns), cluster = clusters))

# Split the data into training and test sets
set.seed(123)
train_index <- sample(seq_len(nrow(mydata)), 0.8 * nrow(mydata))
train_data <- mydata[train_index, ]
test_data <- mydata[train_index, ]

# Alternative method for splitting data using sample and setdiff
train_index <- sample(seq_len(nrow(mydata)), 0.8 * nrow(mydata))
test_index <- setdiff(seq_len(nrow(mydata)), train_index)
mydata_after_test <- mydata[-test_index, ]
mydata_after_train <- mydata[-train_index, ]

# Train a Lasso Regression model using glmnet
train_index <- sample(seq_len(nrow(mydata)), 0.8 * nrow(mydata))
train_data <- mydata[train_index, ]
test_data <- mydata[-train_index, ]
numeric_columns <- dplyr::select(train_data, where(is.numeric))
lasso_model <- glmnet(as.matrix(numeric_columns), train_data$Age, alpha = 1)
summary(lasso_model)
plot(lasso_model, xvar = "lambda", label = TRUE)

# Make predictions on the test set and evaluate the model
test_predictions <- predict(lasso_model, newx = as.matrix(dplyr::select(test_data, where(is.numeric))))
rsquared <- 1 - sum((test_data$Age- test_predictions)^2) / sum((test_data$Age - mean(test_data$Age))^2)
mse <- mean((test_data$Age - test_predictions)^2)
cat("R-squared:", rsquared, "\n")
cat("Mean Squared Error:", mse, "\n")

# Plot the Lasso Regression path
plot(lasso_model, xvar = "lambda", label = TRUE)


# Plot the Lasso Regression path with legend
plot(lasso_model, xvar = "lambda", main = "Lasso Path", col = 1:ncol(lasso_model$beta))
legend("topright", legend = colnames(lasso_model$beta), col = 1:ncol(lasso_model$beta), lty = 1, cex = 0.8)
