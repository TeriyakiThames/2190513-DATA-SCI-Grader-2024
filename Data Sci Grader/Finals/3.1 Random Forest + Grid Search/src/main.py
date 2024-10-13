# import your other libraries here
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


class MushroomClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def Q1(self):
        """
        1. (From step 1) Before doing the data prep., how many "na" are there in "gill-size" variables?
        """
        return self.df["gill-size"].isna().sum()

    def Q2(self):
        """
        2. (From step 2-4) How many rows of data, how many variables?
        - Drop rows where the target (label) variable is missing.
        - Drop the following variables:
        'id','gill-attachment', 'gill-spacing', 'gill-size','gill-color-rate','stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring-rate','stalk-color-below-ring-rate','veil-color-rate','veil-type'
        - Examine the number of rows, the number of digits, and whether any are missing.
        """
        # Drop unnecessary columns
        self.df = self.df.drop(
            [
                "id",
                "gill-attachment",
                "gill-spacing",
                "gill-size",
                "gill-color-rate",
                "stalk-root",
                "stalk-surface-above-ring",
                "stalk-surface-below-ring",
                "stalk-color-above-ring-rate",
                "stalk-color-below-ring-rate",
                "veil-color-rate",
                "veil-type",
            ],
            axis=1,
        )
        # Drop missing values for "label"
        self.df = self.df.dropna(subset=["label"])
        return self.df.shape

    def Q3(self):
        """
        3. (From step 5-6) Answer the quantity class0:class1
        - Fill missing values by adding the mean for numeric variables and the mode for nominal variables.
        - Convert the label variable e (edible) to 1 and p (poisonous) to 0 and check the quantity. class0: class1
        """
        self.Q2()
        # Impute numerical values
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        # Impute categorical values
        not_target = self.df.columns.difference(["label"])
        self.df[not_target] = self.df[not_target].fillna(
            self.df[not_target].mode().iloc[0]
        )
        # Map e to 1 and p to 0
        self.df["label"] = self.df["label"].map({"e": 1, "p": 0})
        return self.df["label"].value_counts()

    def Q4(self):
        """
        4. (From step 7-8) How much is each training and testing sets
        - Convert the nominal variable to numeric using a dummy code with drop_first = True.
        - Split train/test with 20% test, stratify, and seed = 2020.
        """
        self.Q3()
        # Convert nominal variable to dummy code
        not_target = self.df.drop(columns=["label"])
        nominal_df = not_target.select_dtypes(include=["object"])
        self.df = pd.get_dummies(self.df, columns=nominal_df.columns, drop_first=True)

        # Train test split
        X = self.df.drop(columns=["label"])
        y = self.df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=2020
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return X_train.shape, X_test.shape

    def Q5(self):
        """
        5. (From step 9) Best params after doing random forest grid search.
        Create a Random Forest with GridSearch on training data with 5 CV with n_jobs=-1.
        - 'criterion':['gini','entropy']
        - 'max_depth': [2,3]
        - 'min_samples_leaf':[2,5]
        - 'N_estimators':[100]
        - 'random_state': [2020]
        """
        # Pull info from previous question
        self.Q4()
        # Grid search
        grid_search = GridSearchCV(
            # Model
            estimator=RandomForestClassifier(),
            # Parameters
            param_grid={
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 3],
                "min_samples_leaf": [2, 5],
                "n_estimators": [100],
                "random_state": [2020],
            },
            cv=5,
            n_jobs=-1,
        )

        # Train and find the best parameters
        grid_search.fit(self.X_train, self.y_train)
        best_parameters = grid_search.best_params_
        return best_parameters

    def Q6(self):
        """
        5. (From step 10) What is the value of macro f1 (Beware digit !)
        Predict the testing data set with confusion_matrix and classification_report,
        using scientific rounding (less than 0.5 dropped, more than 0.5 then increased)
        """
        # Pull info from previous question
        best_parameters = self.Q5()

        # Train data using RFC with our best parameters
        dtree = RandomForestClassifier(**best_parameters)
        dtree.fit(self.X_train, self.y_train)
        predictions = dtree.predict(self.X_test)
        return classification_report(self.y_test, predictions)


# Given main()
def main():
    hw = MushroomClassifier("mushroom2020_dataset.csv")  # your csv path
    exec(input().strip())


# # Testing main()
# def main():
#     hw = MushroomClassifier("Data Sci Grader/Finals/3.1 Random Forest + Grid Search/src/data/mushroom2020_dataset.csv")
#     print(hw.Q1())

if __name__ == "__main__":
    main()
