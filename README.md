# San-Francisco-crimes

## Introduction
Goal of this competition hosted by kaggle,

https://www.kaggle.com/c/sf-crime

is to predict the type of crimes occurring at a particular date, time and location in San Francisco. The data contain actual incidents and range from January 1, 2003 to May 13, 2015. The data have been split in such a way that the training set consists of the odd numbered weeks, while the test data contains the even numbered weeks.

We are given the following data fields:

- **Dates:** timestamp of the crime incident
- **Category:** category of the crime incident (only in train.csv). This is the target variable we are predicting
- **Descript:** detailed description of the crime incident (only in train.csv)
- **DayOfWeek:** the day of the week
- **PdDistrict:** name of the Police Department District
- **Resolution:** how the crime incident was resolved (only in train.csv)
- **Address:** the approximate street address of the crime incident 
- **X:** Longitude
- **Y:** Latitude

## Analyzing and cleaning the data
This part was performed with `AnalyzeAndClean.py`

#### Multiple Crimes











