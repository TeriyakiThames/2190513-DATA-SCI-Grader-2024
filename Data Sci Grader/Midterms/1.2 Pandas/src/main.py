import json

import pandas as pd

vdo_df = pd.read_csv("Data Sci Grader/Midterms/1.2 Pandas/src/data/US_category_id.json")


"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
"""


def Q1():
    """
    1. How many rows are there in the GBvideos.csv after removing duplications?
    - To access 'GBvideos.csv', use the path '/data/GBvideos.csv'.
    """
    vdo_df = pd.read_csv(
        "Data Sci Grader/Midterms/1.2 Pandas/src/data/US_category_id.json"
    )
    vdo_df = vdo_df.drop_duplicates()
    return int(vdo_df.shape[0])


def Q2(vdo_df):
    """
    2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    filtered = vdo_df[vdo_df["dislikes"] > vdo_df["likes"]]
    return filtered["title"].nunique()


def Q3(vdo_df):
    """
    3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    """
    comments_condition = vdo_df.comment_count > 10000
    date_condition = vdo_df.trending_date == "18.22.01"
    return int(vdo_df[comments_condition & date_condition].shape[0])


def Q4(vdo_df):
    """
    4. Which trending date that has the minimum average number of comments per VDO?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    trending_date = vdo_df.groupby("trending_date")
    comments_per_day = trending_date["comment_count"].mean()
    return str(comments_per_day.idxmin())


def Q5(vdo_df):
    """
    5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - You must load the additional data from 'GB_category_id.json' into memory before executing any operations.
        - To access 'GB_category_id.json', use the path '/data/GB_category_id.json'.
    """
    sports_id = 17
    comedy_id1 = 23
    comedy_id2 = 34
    sports_df = (
        vdo_df[vdo_df["category_id"] == sports_id]
        .groupby("trending_date")["views"]
        .sum()
    )
    comedy_df = (
        vdo_df[vdo_df["category_id"] == (comedy_id1 or comedy_id2)]
        .groupby("trending_date")["views"]
        .sum()
    )
    combined_df = pd.DataFrame(
        {"sports_views": sports_df, "comedy_views": comedy_df}
    ).fillna(0)
    sports_morethan_comedy = (
        combined_df["sports_views"] > combined_df["comedy_views"]
    ).sum()
    return int(sports_morethan_comedy)


#  Run to check if it works!
# def check():
#     print(Q1()==40901 and type(Q1())==int)
#     print(Q2(vdo_df)==122 and type(Q2(vdo_df))==int)
#     print(Q3(vdo_df)==28 and type(Q3(vdo_df))==int)
#     print(Q4(vdo_df)=='18.05.02' and type(Q4(vdo_df))==str)
#     print(Q5(vdo_df)==83 and type(Q5(vdo_df))==int)

# check()
