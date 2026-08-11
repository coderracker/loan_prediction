To download the data, download csv file from https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

EDA: I performed EDA. the data was clean, but there were some modifications. I found that cibil_score has the highest correlation with loan status.

Preprocessing: This project uses a simple preprocessing workflow before training the model. 
First, the loan dataset is loaded and cleaned by fixing column names and standardizing text values such as loan status, education, and self-employment. 
Then, any negative or missing asset values are handled by replacing them with zero so the data stays usable. 
Next, a few new financial features are created, such as loan-to-income ratio and total assets, to help the model learn better patterns. 
Finally, the data is split into training and testing sets so the model can be evaluated properly.

pipeline: I built a simple machine learning pipeline that combines preprocessing and model training in one flow. It prepares the input data, handles categorical and numeric features, and then trains a Random Forest classifier to predict loan approval. This makes the workflow cleaner and easier to reuse for testing different models.

I tried Logistic regression with n = 1000 and my accuracy score was 0.905152224824356. Since I wanted to test other models first to find the most accurate, I tried random_forest. with n_estimators = 100, my accuracy score was 1, taking the model 4 second to train. I tried with n_estimators = 50, it took 2.7 sec with accuracy score of 1, precsion 1, recall 1, f1 score 1. I tried with n_estimators = 10, it was still 1. I tried it with 3, accuracy score was 0.9929742388758782, so the best tuning of hyperparameter, is n_estimators = 7.

Afterwards I saved the model in models folder using joblib.

