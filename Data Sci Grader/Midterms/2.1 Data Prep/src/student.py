import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic.csv) and answer the questions.
"""

def Q1(df):
    """
        Problem 1:
            How many rows are there in the “titanic.csv?
    """
    return df.shape[0]

def Q2(df):
    '''
        Problem 2:
            Drop unqualified variables
            Drop variables with missing > 50%
            Drop categorical variables with flat values > 70% (variables with the same value in the same column)
            How many columns do we have left?
    '''
    temp_df = df.copy()

    missing_threshold = 0.5*len(temp_df)
    temp_df.dropna(thresh=missing_threshold, axis=1)

    flat_value_threshold = 0.7
    for col in temp_df.columns:
        top_freq = temp_df[col].value_counts(normalize=True).max()
        if top_freq > flat_value_threshold:
            temp_df.drop(col, axis=1, inplace=True)
    return temp_df.shape[1]

def Q3(df):
    '''
        Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    temp_df = df.copy()
    temp_df.dropna(subset="Survived",inplace=True)
    return temp_df.shape[0]

def Q4(df):
    '''
        Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    temp_df = df.copy()
    # Q1 = df['Fare'].quantile(0.25)
    # Q3 = df['Fare'].quantile(0.75)
    Q3, Q1 = np.percentile(temp_df["Fare"],[75,25])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    higher_bound = Q3 + 1.5*IQR

    temp_df.loc[df["Fare"] < lower_bound, "Fare"] = lower_bound
    temp_df.loc[df["Fare"] > higher_bound, "Fare"] = higher_bound
    return round(temp_df["Fare"].mean(),2)

def Q5(df):
    '''
        Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    temp_df = df.copy()
    df.fillna(temp_df.mean(numeric_only=True),inplace=True)
    return round(temp_df.Age.mean(),2)

def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    temp_df = df.copy()
    temp_df["Embarked_Q"] = np.where(temp_df["Embarked"] == "Q", 1, 0)
    temp_df["Embarked_C"] = np.where(temp_df["Embarked"] == "C", 1, 0)
    temp_df["Embarked_S"] = np.where(temp_df["Embarked"] == "S", 1, 0)
    return round(temp_df["Embarked_Q"].mean(),2)

def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection
    '''
    temp_df = df.copy().copy()
    temp_df.dropna(subset=["Survived"], inplace=True)
    y = temp_df.pop("Survived")
    x = temp_df
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, stratify=y)
    return round((y_train == 1).mean(), 2)

#  Run to check if it works!
# def check():
#     df = pd.read_csv('Data Sci Grader/2.1 Data Prep/src/data/titanic_to_student.csv', index_col=0)
#     print(Q1(df)==445)
#     print(Q2(df)==10)
#     print(Q3(df)==432)
#     print(Q4(df)==26.27)
#     print(Q5(df)==29.14)
#     print(Q6(df)==0.06)
#     print(Q7(df)==0.41)

# check()