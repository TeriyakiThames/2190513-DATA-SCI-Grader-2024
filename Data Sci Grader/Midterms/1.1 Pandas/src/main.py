import pandas as pd


def main():
    # file = input()
    data = pd.read_csv(
        "Data Sci Grader/Midterms/1.1 Pandas/src/data/scores_student.csv"
    )
    action = input()

    if action == "Q1":
        # Show rows and column of the data frame
        print(data.shape)

    elif action == "Q2":
        # Shows the maximum value of the 'score' column
        print(data["score"].max())

    elif action == "Q3":
        # Filter the data frame for score that is >= 80
        # Then show the number of rows
        condition = data["score"] >= 80
        print(data[condition].shape[0])

    else:
        print("No Output")


if __name__ == "__main__":
    main()
