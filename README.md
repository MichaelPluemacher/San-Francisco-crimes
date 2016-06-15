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
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/MultCrime_by_Cat.png)
On the x axes we have the number of crimes in one event. It is immediately obvious that certain types of crimes occur almost exclusively in mutiple events, while others prefer to happen in isolation. Another way of looking at it is to plot the probabilty of crimes by their category for the different values of the number of crimes in one event:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/MultCrime_by_Count.png)
Again, we see significant differences in the distributions for crimes occuring in isolation (*MultCount*=1) compared to those occuring simultaneously with other crimes. We take that into consideration by computing the log odds of a crime occuring for the different values of *MultCount*. Given the scarcity of data for *MultCount*>2 we only consider *MultCount*=1, 2, and larger than or equal to 3 separately. Those condensed probabilities are summarized in the following graph:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/MultCrime_by_Count_condensed.png)
We add the corresponding log odds both to the train and test datasets

#### The timestamp
The datetime stamp provided in the data is split into the following distinct columns:
 - Year,
 - Month,
 - Day of the month,
 - Ordinal day of the year,
 - Time (hour + minute/60),
 - Hour,
 - and minute.

We check that there are no data for February 29 in the data, i.e. leap years are one problem we don't have to worry about here. 

Further, we check whether daylight saving time is implemented in the times given or not. For example, in 2014, DST started on March 9 at 2 am, i.e. there should be no data between 2am and 3am on this day. However, there are such data, hence DST seems not to be implemented here, yet another complication we don't have to worry about.

Additionally, we introduce a new column denoting whether it is night or not. As a (very) rough divider we use the average time of sunrise and sunset per month.

Presumably, crime rates will be different on working days on the one hand and weekdays and holidays on the other hand. So we'll introduce a column *WorkingDay* to differentiate accordingly. Data on holidays and which businesses actually observe them are sketchy at best, so we'll only count the most important ones as holidays:
  - New Year
  - Memorial Day
  - Independence Day
  - Labor Day
  - Thanksgiving
  - Black Friday
  - Christmas

Technically, Black Friday is not a holiday, of course. However, we include it here due to its special significance. The corresponding holiday rules have been implemented with the help of the *USFederalHolidayCalendar* from *pandas.tseries.holiday*

#### The crimes
Time to look at the different crimes. Counting the numbers by categories gives:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/CrimeCounts.png)
We also look at the distributions of crimes over the day and for different days of the week. As an example let us show those distributions for violent crimes:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/violentCrimes_hour.png)
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/violentCrimes_day.png)
As we can see there is a fairly broad distribution over the course of the day and evening with a significant dip in the early hours. The distribution over the week seems to peak on Wednesdays.

As another interesting example, let's look at economic crimes:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/economicCrimes_hour.png)
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/economicCrimes_day.png)
In the hourly distribution we notice two peaks at 12am and 12pm. From the description of the data it is not immediately obvious why that should be the case. Maybe it's because such crimes tend to go on over a period of time, i.e. no single time can be assigned to them.

#### The addresses
Following an idea first put forward by papadopc on kaggle

https://www.kaggle.com/papadopc/sf-crime/neural-nets-and-address-featurization

we turn the addresses into a feature by computing the log odds of an address occuring in the data sets as well as the log odds of a particular crime occuring at that address.

First, however, we have to do some cleaning on the addresses. We start by differentiating between intersections of two streets and regular street addresses, i.e. we introduce a column *StreetCorner* indicating whether it is an intersection or not.

Further, the raw data contain plenty of misspellings and identical entries not recognizable as such, e.g. *A ST / B ST* and *B ST / A ST* are clearly the same intersection. We do that by extracting the street names, removing suffixes such as *ST*, *AV*, *HWY* etc. and then combine them in alphabetical order for intersections. Having done that, we compute the log odds as briefly described above.

#### Latitude and Longitude
Finally, the geographical coordinates. Some values are clearly wrong. Indeed, we have 67 entries in train and 76 in test where the latitude is given as 90 degrees, i.e. the north pole. As those mistakes are mercifully few, we just replace these values by the medians for the corresponding police district.

Looking at the geographical distribution of various crimes, one notices clear hotspots. As an example, let's consider weapons laws violations. A scatter plot overlaid on a map of San Francisco:
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/WeaponLaws_scatter.png)
And a corresponding heat map
![alt text](https://github.com/MichaelPluemacher/San-Francisco-crimes/blob/master/Graphs/WeaponLaws_heat.png)
Clearly, the coordinates play an important role in predicting crimes. Hence, we incorporate the following transformed coordinates into the data:
- X and Y rotated by 30, 45 and 60 degrees,
- and the distance from the center of the map.

In the end, only the distance plays any role in fitting models. Clearly, a more refined construction of features based on X and Y could be useful. For example, as we saw above, crimes tend bo be clustered around hotspots. Computing the distance from the center of the hotspot separately for each category of crime might therefore prove useful.

## Building a predictive model
After all the feature engineering we end up with 89 numerical predictors. This is quite a lot and since most of them were derived from each other, one would expect strong correlations, at least between some of them. As a first attempt to deal with that we perform a principal component analysis (PCA) followed by a dimensionality reduction. However, this slightly weakens the predictivity of our models. Second, one can check which of those variables are most important in building models and drop most or at least some of the unimportant ones. Again, we tried that and found that it worsens the performance of models. Hence, we use all predictors in our attempt to predict crime in San Francisco. The factor variables left in the data, i.e. *PdDistrict* and *DayOfWeek*, are label-encoded with the preprocessor of the *sklearn* package.

Now for the fun part, building models! We start by training various models included in *sklearn*. Here, only *RandomForestClassifier*, *ExtraTreesClassifier*, and *GradientBoostingClassifier* perform decently, giving a log loss of about 2.3 to 2.4 on the public leaderboard. Next we turn our attention to some models from the *h2o* package. The *H2ORandomForestEstimator* performs about as well as the random forest from *sklearn*. To our surprise, the deep learning neural network *H2ODeepLearningEstimator* performs rather poorly. A more careful tuning of parameters and selection of predictors may still yield decent results here.

In the end, our best model is an *XGBClassifier* from the *xgboost* package. This is implemented in `XGBoost_model.py` and gives a log loss of 2.278 on the public leaderboard. Not particularly brilliant, but not all that bad either.
