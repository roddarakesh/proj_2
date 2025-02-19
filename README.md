# Valentine's Day Gift Recommendation Using Machine Learning

### Table of Contents

1. [Contributors](#contributors)
2. [Executive Summary](#executive-summary)
3. [Project Objectives](#project-objectives)
4. [Research Approach](#research-approach)
5. [Datasets](#datasets)
6. [Analysis and Conclusion](#analysis-and-conclusion)
7. [Demo](#demo)
8. [Next Steps](#next-steps)
9. [Tools and Libraries](#tools-and-libraries)
10. [How To Run](#how-to-run)

---

### Contributors
![Contributors' photos](backup/team-picture.pptx.png)

---

### Executive Summary

This project aims to develop a machine learning model to assist individuals in selecting a Valentine's Day gift based on various recipient characteristics. Despite extensive research, no suitable publicly available dataset was found. Therefore, a synthetic dataset was generated using ChatGPT and refined through multiple iterations. The project explores the feasibility of using machine learning for personalized gift recommendations and evaluates the performance of various classification models.

The ultimate goal is to provide a tool that enhances the gift-giving experience by leveraging data-driven insights. While the initial results were below expectations, the project lays the groundwork for future improvements, including dataset refinement, advanced modeling techniques, and the development of a user-friendly interface.

---

### Project Objectives

#### 1. **Dataset Creation**
   - Generate a synthetic dataset using ChatGPT to simulate recipient characteristics and gift preferences.
   - Refine the dataset through multiple iterations to improve data quality and relevance.

#### 2. **Data Preprocessing**
   - Clean and format the dataset to ensure consistency.
   - Handle missing values and encode categorical variables for machine learning compatibility.

#### 3. **Model Development**
   - Experiment with classification models, including Decision Trees and Random Forests.
   - Train and evaluate models using Scikit-learn.

#### 4. **Model Evaluation**
   - Assess model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Analyze feature importance to identify key predictors of gift preferences.

#### 5. **Insights and Recommendations**
   - Provide actionable insights for improving the model and dataset.
   - Explore potential applications and future directions for the project.

---

### Research Approach

A machine learning approach was employed, focusing on classification models to predict the best gift based on recipient characteristics. The project involved dataset generation, preprocessing, model training, and evaluation. Descriptive and diagnostic analyses were conducted to understand the dataset and model performance.

---

### Datasets

#### 1. **Synthetic Valentine's Day Gift Dataset**
   - **Description**: A synthetic dataset containing 2000 records with features such as recipient gender, age, relationship status, budget, personal interests, and past gift reactions. The target variable is the best gift recommendation.
   - **Source**: Generated using ChatGPT and refined through multiple iterations.

---

### Analysis and Conclusion

The analysis revealed several key insights:

1. **Dataset Limitations**: The synthetic dataset, while useful for initial experimentation, did not yield highly accurate predictions. Initial model accuracy was around **17%**, improving to just under **60%** in later iterations, and culminating in a final version with an accuracy of **97%**.
   
2. **Model Performance**: Classification models such as Decision Trees and Random Forests showed moderate performance initially, indicating a need for more robust data or advanced modeling techniques. Optimization yields feeble results on such poorly-performing models, so we returned to the data. After further data refinement, focusing on pattern generation and trainable redundancy, an eventual accurancy score of **97%** was reached. 

3. **Feature Importance**: Certain features, such as **Relationship Status** and **Personal Interest**, had a significant impact on predictions, while others, like **Gift Popularity**, were less influential.

4. **Future Potential**: Despite the limitations, the project demonstrated the feasibility of using machine learning for personalized gift recommendations. Future work could focus on refining the dataset, exploring deep learning approaches, and developing a user-friendly interface.

---

### Demo

Watch the demo video [here](https://vimeo.com/1058422509/26c8057dd0?share=copy).

[![Video Thumbnail](backup/demo-thumbnail.png)](https://vimeo.com/1058422509/26c8057dd0?share=copy)

---

### Next Steps

1. **Dataset Refinement**: Collect real-world data to improve model applicability.
2. **Advanced Modeling**: Experiment with deep learning models, such as neural networks, to enhance predictive performance.
3. **Feature Engineering**: Identify and incorporate additional features that may influence gift preferences.
4. **User Interface Development**: Create a user-friendly application for real-time gift recommendations.
5. **Collaboration**: Partner with gift retailers or survey platforms to gather more comprehensive data.

---

### Tools and Libraries

The following tools and libraries were used for this project:

- **Python**: Primary programming language for data processing and model development.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning model training and evaluation.
- **Matplotlib & Seaborn**: Data visualization.
- **Google Colab**: Interactive development and documentation.
- **ChatGPT**: Synthetic dataset generation.

---

### How To Run

1. **Prepare the Dataset**  
   - Download the [dataset](data/valentine_gift.csv).
      
2. **Open the Application**  
   - Launch Google Colab.

3. **Run the Notebook**  
   - Open the notebook file.
   - Import Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, [dataset](data/valentine_gift.csv). 
   - Run all the cells. 

 
