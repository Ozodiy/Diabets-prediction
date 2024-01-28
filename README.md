# Diabetes Data Analysis

## Description

**Diabetes Data Analysis** is a comprehensive analysis project focusing on the prediction and understanding of diabetes in individuals. Utilizing a robust dataset with multiple diagnostic measurements, this project applies advanced data science techniques to explore, visualize, and model the data. Our goal is to uncover hidden patterns and insights that can lead to a better understanding of diabetes indicators and eventually aid in predicting diabetes onset with significant accuracy.

### Objectives

- To understand the relationships and patterns within the diabetes data.
- To develop predictive models that can accurately classify individuals' diabetes status.
- To provide a detailed exploratory data analysis (EDA) with visualizations that highlight key factors related to diabetes.

## Data

The dataset contains the following features:

- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skinfold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age (years)
- `Outcome`: Class variable (0 or 1)

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms

## Methodology

The project follows the OSEMN pipeline:

1. **Obtain**: The data is sourced from a reputable medical data repository.
2. **Scrub**: Cleaning and preprocessing the data to ensure quality analysis.
3. **Explore**: Conducting exploratory data analysis (EDA) to understand the data and find patterns.
4. **Model**: Applying various machine learning models to predict diabetes onset.
5. **iNterpret**: Interpreting the results, understanding the model performance, and deriving insights.

### Exploratory Data Analysis

In-depth analysis including statistical summaries, distribution of features, correlation analysis, and other visualizations to understand the data deeply.

### Predictive Modeling

Several machine learning models were experimented with, including Logistic Regression, SVM, Random Forest, and Gradient Boosting, to find the most accurate and reliable model.

## Streamlit Application

Our project includes an interactive web application built with Streamlit. You can visualize data, interact with the analysis, and input data for predictions.

**Access the App**: [Streamlit App](http://192.168.0.16:8502)
