# Valentine's Day Gift Recommendation Using Machine Learning

### Table of Contents

1. [Contributors](#contributors)
2. [Executive Summary](#executive-summary)
3. [Project Objectives](#project-objectives)
4. [Research Approach](#research-approach)
5. [Datasets](#datasets)
6. [Analysis and Conclusion](#analysis-and-conclusion)
7. [Next Steps](#next-steps)
8. [Tools and Libraries](#tools-and-libraries)
9. [How To Run](#how-to-run)
10. [License](#license)

---

### Contributors

Gabe Galley, Leonard Forrester, Leslie Bland, Rakesh, Sophak So, Yujing Li, and TA Deborah Aina

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

1. **Dataset Limitations**: The synthetic dataset, while useful for initial experimentation, did not yield highly accurate predictions. Initial model accuracy was around **17%**, improving to just under **60%** in later iterations.
   
2. **Model Performance**: Classification models such as Decision Trees and Random Forests showed moderate performance, indicating a need for more robust data or advanced modeling techniques.

3. **Feature Importance**: Certain features, such as **Relationship_Status** and **Personal_Interest**, had a significant impact on predictions, while others, like **Gift_Popularity**, were less influential.

4. **Future Potential**: Despite the limitations, the project demonstrated the feasibility of using machine learning for personalized gift recommendations. Future work could focus on refining the dataset, exploring deep learning approaches, and developing a user-friendly interface.

---

### Next Steps

1. **Dataset Refinement**: Collect real-world data or further refine the synthetic dataset to improve model accuracy.
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

2. **Prepare the Dataset**  
   - Download the dataset.   

3. **Open the Application**  
   - Launch the Google Colab.  

4. **Run the Notebook**  
   - Open the notebook file.
   - Run all the cells. 

 
