#Project 2
# Valentine's Day Gift Prediction Project

## Project Overview

This project aims to predict the best Valentine's Day gift based on various factors such as age, gender, relationship status, and past gift preferences. It utilizes a machine learning approach to classify gifts into categories like flowers, jewelry, gadgets, etc.

## Dataset

The project utilizes the `best_valentine_gift_dataset.csv` dataset, which contains information about individuals and their preferred gifts. The dataset includes the following features:

- **Age:** Recipient's age
- **Gender:** Recipient's gender
- **Relationship_Status:** Relationship status of the giver and recipient
- **Past_Gift:** Type of gift given in the past
- **Best_Gift:** The most preferred gift category

## Methodology

The project follows these steps:

1. **Data Loading and Preprocessing:**
   - The dataset is loaded using the pandas library.
   - Categorical features are encoded using one-hot encoding.
   - The target variable ("Best_Gift") is label encoded.

2. **Correlation Analysis:**
   - A correlation matrix is calculated to identify relationships between numerical features.
   - The correlation matrix is visualized using a heatmap.

3. **Model Training and Evaluation:**
   - Various classification models, including Random Forest, Decision Tree, SVC, and AdaBoost, are trained on the preprocessed data.
   - Model performance is evaluated using metrics such as accuracy and R-squared.

4. **Model Optimization:**
   - Hyperparameter tuning is performed using GridSearchCV to optimize model performance.

## Results

The results of the model training and evaluation are presented in the notebook. The best-performing model is selected based on its accuracy and R-squared value. Unfortunately, despite attempting to use various datasets, all three did not provide satisfactory results

## Conclusion

This project demonstrates the application of machine learning techniques to predict the best Valentine's Day gift. The developed model can be used to assist individuals in selecting the most suitable gift for their loved ones.

## Future Work

Potential future improvements include:

- Seeking an improved dataset capable of delivering a higher predictive accuracy score.
- Exploring other advance machine learning algorithms.
- Incorporating additional features into the dataset.
- Deploying the model for real-time predictions.


