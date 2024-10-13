# import your other libraries here
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Clustering:
    def __init__(self, file_path):
        # Add other parameters if needed
        self.file_path = file_path
        self.df = None
        self.scaler = None
        self.centroid = None  # Also called Cluster Centers

    def Q1(self):
        """
        1. Load the CSV file.
        2. Choose edible mushrooms only.
        3. Only the variables below have been selected to describe the distinctive
            characteristics of edible mushrooms:
            'cap-color-rate','stalk-color-above-ring-rate'
        4. Provide a proper data preprocessing as follows:
            - Fill missing with mean
            - Standardize variables with Standard Scaler
        """
        # Load CSV
        self.df = pd.read_csv(self.file_path)
        edible_df = self.df[self.df["label"] == "e"]

        # Select only edible mushrooms
        selected_columns = ["cap-color-rate", "stalk-color-above-ring-rate"]
        edible_df = edible_df[selected_columns]

        # Imputing missing values with mean
        edible_df = edible_df.fillna(edible_df.mean(numeric_only=True))

        # Standardize variables
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(edible_df)
        self.df = scaled_data
        return scaled_data.shape

    def Q2(self):
        """
        5. K-means clustering with 5 clusters (n_clusters=5, random_state=0,
            n_init='auto')
        6. Show the maximum centroid of 2 features ('cap-color-rate' and
            'stalk-color-above-ring-rate') in 2 digits.
        """
        self.Q1()
        # Set K Means parameters
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        kmeans.fit(self.df)

        # Get centroid from K Means
        self.centroid = kmeans.cluster_centers_
        max_centroid = self.centroid.max(axis=0)
        return np.round(max_centroid, 2)

    def Q3(self):
        """
        7. Convert the centroid value to the original scale, and show the
            minimum centroid of 2 features in 2 digits.
        """
        self.Q2()
        # Get StandardScaler and inverse transform the centroid
        original_scale = self.scaler.inverse_transform(self.centroid)
        min_centroid = original_scale.min(axis=0)
        return np.round(min_centroid, 2)


def main():
    hw = Clustering(
        "Data Sci Grader/Finals/3.3 K-Means/src/data/ModifiedEdibleMushroom.csv"
    )
    print(hw.Q2())


if __name__ == "__main__":
    main()
