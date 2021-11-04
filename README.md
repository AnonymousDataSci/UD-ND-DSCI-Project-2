# Disaster Response Pipeline Project

## Project Description
This projects aims to built a model that classifies messages during a desaster. Examples are provided by figure8. There are different categories a disaster message can be part of. Goal is to train a model that can classify unseen messages in one or more of the catergories

## File Structure
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- train_classifier.py
                
## Was was done
- First an ETL pipeline was build to read in data from two different csv files
- Second the Pipeline cleans the data and saves it in a SQL Database
- Third the cleaned data was used to train a classifer
- Last a Flask app was provided with data visualization that can classify a message

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
