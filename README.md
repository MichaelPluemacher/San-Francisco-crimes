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
Occasionally, multiple crimes occur at the same location, date and time. Those are probably related, i.e. one evil deed consisting of several crimes. We take that into account by adding a new column with the number of simultaneous crimes (*MultCount*) and by computing the log odds of crimes occurring simultaneously. Counting the number of such occurences in the test data yields:

| *MultCount* | count |
|---------------:|------:|
| 1   |   557717| 
| 2   |    97664| 
| 3   |    33855| 
| 4   |     4265| 
| 5   |     1392| 
| 6   |      468| 
| 7   |      180| 
| 8   |       87| 
| 9   |       40| 
| 10  |       23| 
| 11  |        9| 
| 12  |        4| 
| 13  |        2| 
| 14  |        1| 
| 19  |        1| 
| 32  |        1| 
| 40  |        1| 

Similarly, in the training data we have:

| *MultCount* | count |
|---------------:|------:|
| 1   |   550477| 
| 2   |    97879| 
| 3   |    34002| 
| 4   |     4358| 
| 5   |     1392| 
| 6   |      467| 
| 7   |      165| 
| 8   |       80| 
| 9   |       39| 
| 10  |       18| 
| 11  |       11| 
| 12  |        6| 
| 13  |        5| 
| 14  |        1| 
| 16  |        1| 

Hence, events with more that three crimes occurring simultaneously are comparatively rare. We have 39 distinct categories of crimes in the data. Let's look at the probability of a particular type of crime being part of a multiple event:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/MultiCrime_by_Cat.png)
On the x axes we have the number of crimes in one event. It is immediately obvious that certain types of crimes occur almost exclusively in mutiple events, while others prefer to happen in isolation. Another way of looking at it is to plot the probabilty of crimes by their category for the different values of the number of crimes in one event:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/MultiCrime_by_Count.png)
Again, we see significant differences in the distributions for crimes occuring in isolation (*MultCount*=1) compared to those occuring simultaneously with other crimes. We take that into consideration by computing the log odds of a crime occuring for the different values of *MultCount*. Given the scarcity of data for *MultCount*>2 we only consider *MultCount*=1, 2, and larger than or equal to 3 separately. Those condensed probabilities are summarized in the following graph:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/MultiCrime_by_Count_condensed.png)
We add the corresponding log odds both to the train and test datasets

#### The timestamp
















