# Comparative Analysis for Heart Disease Prediction

## Objective

The goal of this project is to identify the best predictive model for heart disease risk. We aim to evaluate different machine learning models to find the one that best predicts heart disease, enabling its use in screening high-risk populations.

## Libraries and Tools

The project utilizes the following libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib** and **seaborn**: For data visualization.
- **scikit-learn**: For machine learning algorithms and model evaluation.
- **KneeLocator**: For identifying the optimal number of clusters in K-Means clustering.
- **LogisticRegression, KMeans, KNeighborsClassifier, svm**: For model building and evaluation.

## Dataset

We use a heart disease dataset containing 303 rows and 14 columns. Each row represents a patient, and the columns contain medical attributes and the target variable (whether the patient has heart disease or not).

### Data Preview:

| age | sex | cp | trtbps | chol | fbs | restecg | thalachh | exng | oldpeak | slp | caa | thall | output |
|-----|-----|----|--------|------|-----|---------|----------|------|---------|-----|-----|-------|--------|
| 63  | 1   | 3  | 145    | 233  | 1   | 0       | 150      | 0    | 2.3     | 0   | 0   | 1     | 1      |
| 37  | 1   | 2  | 130    | 250  | 0   | 1       | 187      | 0    | 3.5     | 0   | 0   | 2     | 1      |

The dataset has no missing values, and all variables are either categorical or numerical.

## Exploratory Data Analysis (EDA)

We performed EDA to understand the distribution and characteristics of the data:

1. **Distribution of Numerical Variables**: 
   - Most patients are between 50 and 60 years old.
   - Blood pressure is mostly between 110 and 140.
   - Cholesterol levels are between 200 and 280 for most patients.

2. **Categorical Variables**: 
   - 68.3% of patients are male, and 31.7% are female.
   - 54.5% of the patients are at risk for heart disease.

3. **Heatmap**: 
   - A heatmap was created to show the correlation between the numerical variables. Age and maximum heart rate have the highest negative correlation.

## Preprocessing

- **Categorical Conversion**: Variables such as `sex`, `cp`, `fbs`, `restecg`, `exng`, etc., were converted to categorical types.
- **Data Scaling**: The `StandardScaler` from `sklearn` was used to normalize the numerical features.

## Data Splitting

The dataset was split into training (70%) and testing (30%) sets:
- **X**: Features (independent variables).
- **Y**: Target (dependent variable - output).

```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=4)
```

## Modeling

We used three machine learning algorithms for heart disease prediction:

1. **K-Nearest Neighbors (KNN)**:
   - Optimal number of clusters (`k=4`) was determined using the **Knee Locator** method.
   - **Accuracy**: `71.4%`

2. **Logistic Regression**:
   - A simple linear model used to predict heart disease risk.
   - **Accuracy**: `80.2%` (Best-performing model)

3. **Support Vector Machine (SVM)**:
   - SVM with a linear kernel was trained and tested.
   - **Accuracy**: `79.1%`

### Model Performance Summary

| Model               | Accuracy  |
|---------------------|-----------|
| K-Nearest Neighbors  | 71.4%     |
| Logistic Regression  | 80.2%     |
| Support Vector Machine | 79.1%   |

### Conclusion

The **Logistic Regression** model was the best performer, achieving an accuracy of **80.2%**. This model can be used to effectively predict heart disease risk in the dataset.

## Next Steps

- Hyperparameter tuning to improve model performance.
- Incorporation of additional features or datasets for enhanced prediction accuracy.

## License

This project is licensed under the MIT License.
