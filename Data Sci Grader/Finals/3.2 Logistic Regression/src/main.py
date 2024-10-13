import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BankLogistic:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=",")
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def Q1(self):
        """
        Problem 1:
        Load ‘bank-st.csv’ data from the “Attachment”
        How many rows of data are there in total?
        """
        return self.df.shape[0]

    def Q2(self):
        """
        Problem 2:
        How many numeric variables are there ?
        """
        temp_df = self.df.copy()
        return temp_df.select_dtypes(include="number").shape[1]

    def Q3(self):
        """
        Problem 3:
        How many categorical variables are there?
        """
        temp_df = self.df.copy()
        return temp_df.select_dtypes(include="object").shape[1]

    def Q4(self):
        """
        Problem 4:
        What is the distribution of the target variable for the 'NO' class?
        """
        temp_df = self.df.copy()
        no_count = temp_df[temp_df["y"] == "no"].shape[0]
        total_count = temp_df.shape[0]
        return no_count / total_count

    def Q5(self):
        """
        Problem 5:
        Drop duplication of the data
        What are the shapes of the data?
        """
        self.df = self.df.drop_duplicates()
        return self.df.shape

    def Q6(self):
        """
        Problem 6:
        How many null values are there in the job and education columns?
        For numeric variables, fill missing values with the mean, and for
        categorical variables, fill missing values with the mode.
        Hint: replace unknown with null
        """
        self.Q5()

        # Replace 'unknown' with NA
        self.df.replace("unknown", pd.NA, inplace=True)

        # Count nulls in job and education columns
        null_job = int(self.df["job"].isnull().sum())
        null_education = int(self.df["education"].isnull().sum())

        # Fill missing values for numeric and categorical columns
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        return null_job, null_education

    def Q7(self):
        """
        Problem 7:
        Split train/test for 70%:30% with random_state=0 & stratify option
        What are the shapes of X_train and X_test?
        Hint: Don't forget to encode categorical data using pd.get_dummies before
        splitting the data.
        """
        self.Q6()
        # Create dummy code without "y"
        not_target = self.df.drop(columns=["y"])
        nominal_df = not_target.select_dtypes(include=["object"])
        self.df = pd.get_dummies(self.df, columns=nominal_df.columns, drop_first=True)

        # Train Test Split
        X = self.df.drop("y", axis=1)
        y = self.df["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )

        # Save to self
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return X_train.shape, X_test.shape

    def Q8(self):
        """
        Problem 8:
        How much data does the test set contain for the class 'No'?
        """
        self.Q7()
        return self.y_test[self.y_test == "no"].shape[0]

    def Q9(self):
        """
        Problem 9:
        Set the model to be used as Logistic Regression only with random_state=0
        and max_iter=5000
        Build a model that uses all variables.
        What is the macro F1 score for Model on the test data? (Answer with two
        decimal places).
        """
        self.Q7()
        # Create logistic regression model
        model = LogisticRegression(random_state=0, max_iter=5000)
        model.fit(self.X_train, self.y_train)

        # Create predictions from the model
        predictions = model.predict(self.X_test)
        macro_f1 = f1_score(self.y_test, predictions, average="macro")
        return round(macro_f1, 2)


# **Copy and paste your libraries, class Clustering with modified functions when
# submitting to the grader. (don’t submit the code below)**
def main():
    hw = BankLogistic(
        "Data Sci Grader/Finals/3.2 Logistic Regression/src/data/bank-st.csv"
    )  # your file path
    print(hw.Q1())


if __name__ == "__main__":
    main()
