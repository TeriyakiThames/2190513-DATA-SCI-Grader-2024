import pandas as pd
from student import *

def main():
    input_string = input().strip()
    df = pd.read_csv('Data Sci Grader/2.1 Data Prep/src/data/titanic_to_student.csv', index_col=0)
    if input_string == "Q7":
        df.fillna(df.select_dtypes(
            include='number').mean(), inplace=True)
    input_command = f"{input_string}(df)"
    print(f"{eval(input_command)}")

if __name__ == "__main__":
    main()
