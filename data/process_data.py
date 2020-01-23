# -*- coding: utf-8 -*-
"""
@author: silent-j
process_data.py [module]

"""
import sys
import pandas as pd
from sqlalchemy import create_engine

def process_raw_data(message_path, label_path):
    """
    Func: processes raw text & label data: merges datasets, parses category labels,
          and removes duplicate rows, redundant columns, fixes multiclass labels
    Parameters:
    message_path: str, path .csv containing messages data
    label_path: str, path to .csv containing categories data
    Returns:
    df: pandas dataframe containing the merged and 
        cleaned message & categories data
    """
    # load messages
    messages = pd.read_csv(message_path)
    # load categories dataset
    categories = pd.read_csv(label_path)
    # merge datasets
    df = pd.merge(messages, categories, on='id')

    categories = df['categories'].str.split(';', expand=True)

    row = categories.loc[0]

    category_colnames = [i.split('-')[0] for i in row]
    categories.columns = category_colnames

    to_drop = ['categories']

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        if len(categories[column]) == 1:
            to_drop.append(column)
        if len(categories[column] > 2):
            categories.loc[categories[column]>1, column] = 1

    df = pd.concat([df, categories], axis=1)
    df.drop(to_drop, axis=1, inplace=True)

    print(f"Dataset contains {df.duplicated().sum()} duplicate rows")
    # drop duplicates
    df.drop_duplicates(inplace=True)
    print(f"Dataset contains {df.duplicated().sum()} duplicate rows")
    
    return df

def save_data(df, db_path, table_name):
    """
    Func: save the processed data to a .db file using sql
    Parameters:
    df: pandas dataframe containing cleaned data
    db_path: str, path to save database file to
    table_name: str, name of table in database
    Returns:
    None
    """
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, index=False)

def main():
    if len(sys.argv == 5):       
        message_path, label_path, db_path, table_name = sys.argv[1:]
        
        print(f"Loading data...{message_path} .. {label_path}\n")       
        clean_data = process_raw_data(message_path, label_path)       
        print("Saving cleaned data... {db_path} || {table_name}")       
        save_data(clean_data, db_path, table_name)
        print("Data Processed")
        
if __name__=="__main__":
    main()
    