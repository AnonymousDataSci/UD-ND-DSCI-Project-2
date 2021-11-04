import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data into the Pipeline
    Arguments:
    messages filepath: Filepath to csv data
    catergories filepath : filepath to csv data
    
    Return:
    df: dataframe with combined csv data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df

def clean_data(df):
    """
    Clean Data in the Pipeline
    Arguments:
    df: Dataframe containing message and categories data
    
    Return:
    df: dataframe with cleand data
    """
    #Dataframe with categories in indivdual rows
    categories = df.categories.str.split(pat=';',expand=True)
    
    #Convert the catergories column into numeric values with right descirption
    row = categories.iloc[0]
    category_names = row.apply(lambda x:x[:-2])
    categories.columns = category_names
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
        
    #Add the new categories columns back to the orginal Dataframe and drop duplicate
    df.drop('categories',axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    #print('in der funktion info \n', df)
    return df


def save_data(df, database_filename):
    """
    Save Data in the Pipeline
    Arguments:
    df: Dataframe containing message and categories data
    database filename: Path to DB 
    """
    #print(df.info)
    engine = create_engine('sqlite:///'+ database_filename)
    
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')    
 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        #print('dataframe info \n', df)
        
        print('Cleaning data...')
        df = clean_data(df)
        #print('dataframe info \n', df)
               
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()