# Load necessary libraries
library(tidymodels)
library(tidyverse)
library(kernlab)
library(xgboost)

# Read the training data from "train2.csv" and convert the "action_taken" column to a factor
train <- read_csv("train2.csv")
train$action_taken <- as.factor(train$action_taken)

# Identify columns with missing values
na_cols <- names(train[, colSums(is.na(train)) > 0])
na_only <- names(train[, colMeans(is.na(train)) == 1.0])

# Replace missing values with specific values for selected columns
train[na_cols] <- train[na_cols] %>% 
  replace_na(list(
    ethnicity_of_applicant_or_borrower_1 = 4,
    ethnicity_of_applicant_or_borrower_2 = 4
    # ... (other columns)
  ))

# Select columns without missing values and replace missing values in the other columns with zeros
names(train[, colSums(is.na(train)) > 0])
train_full <- train %>% select(!na_only)
train_zero <- train %>%
  select(na_only) %>%
  replace(is.na(.), 0)

train <- train_full %>% bind_cols(train_zero)

# Set a seed and create cross-validation folds
set.seed(101)
train_folds <- vfold_cv(train, v = 10, strata = action_taken)

# Create a boosting tree model
bt_model <-
  boost_tree(
    mtry = 40,
    trees = 100,
    min_n = 10,
    tree_depth = 10,
    learn_rate = 0.2,
    loss_reduction = 0.01,
    sample_size = 1,
    stop_iter = 5
  ) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

# Define a recipe for data preprocessing
recipe <- 
  recipe(action_taken ~ ., data = train %>% select_if(~!is.character(.)) ) %>%
  step_rm(id) %>% 
  step_impute_mean(all_numeric()) %>%
  step_zv(all_predictors()) %>%
  step_center(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Create a workflow that includes the model and recipe
workflow <-
  workflow() %>%
  add_model(bt_model) %>%
  add_recipe(recipe)

# Fit the model using cross-validation
crossval_fit <- 
  workflow %>% 
  fit_resamples(resamples = train_folds)

# Collect and display model evaluation metrics
crossval_fit %>% collect_metrics()

# Fit the final model using the entire training dataset
final_fit <-
  workflow %>%
  fit(data = train)

# Read the test data from "test2.csv" and handle missing values in a similar way as training data
test <- read.csv("test2.csv")
na_cols <- names(test[, colSums(is.na(test)) > 0])
na_only <- names(test[, colMeans(is.na(test)) == 1.0])
test[na_cols] <- test[na_cols] %>% replace_na(list(
  ethnicity_of_applicant_or_borrower_1 = 4,
  ethnicity_of_applicant_or_borrower_2 = 4
  # ... (other columns)
))
names(test[, colSums(is.na(test)) > 0])

# Select columns without missing values and replace missing values in the other columns with zeros
train_full <- test %>% select(!na_only)
train_zero <- test %>%
  select(na_only) %>%
  replace(is.na(.), 0)

test <- train_full %>% bind_cols(train_zero)

# Make predictions using the final model
predictions <- 
  final_fit %>% 
  predict(new_data = test)

# Combine predictions with the "id" column and write to a CSV file
predictions <-
  test %>% select(id) %>%
  bind_cols(predictions)

write_csv(predictions, "predictions.csv")