### MIE1624H Assignment 2 ##
# Twitter Sentiment Analysis #
#### By: Sanjif Rajaratnam || Student ID: 999091986 || UtorID: rajara24 ####

# The two classifiers that were chosen for the first part of the assignment were
# Naive Bayes and Logistic Regression. The models for this are from the scikit-learn
# libraryThe two classifiers that were chosen for this assignment were the Logistic
# Regression and Naive Bayes model. These models were obtained from the ski-kit learn
# Python libraries. The Multinomial and Bernoulli Naive Bayes model and the Logistic
# Regression model will be analyze in this section. The metrics will also be obtained
# for the scikit-learn library. Visualization will be done with matplotlib.

### Import Libraries ###

# Import libraries for math functions, dataframes, and plotting
import numpy as np # math functions
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting
from matplotlib import style # for plotting style

# Import sci-kit learn libraries
from sklearn.cross_validation import train_test_split # To split data into training/test sets
from sklearn.feature_extraction.text import TfidfVectorizer # To create features from tweets
from sklearn.linear_model import LogisticRegression # Logistic Regression model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB # Naive-Bayes models
from sklearn.metrics import confusion_matrix, classification_report # For model metrics
from sklearn.cross_validation import KFold, cross_val_score # For K-Fold validation
from sklearn import metrics # Standard ML metrics

# Plot style
style.use('ggplot')


## Part 1: Choosing the Optimal Machine Learning Algorithm ##


# The optimal classifier will be chosen by training and testing on the labeled data.
# A Logistic Regression model, a Multinomial Naive-Bayes model, and a Bernoulli Naive-Bayes
# model will be built and have their parameters optimized via grid search. The model
# performance will then be analyzed using the K-Fold technique, and standard statistical
# metrics.

### 1a) Read in Training Data ###

# Change working directory to /resources/data
# %cd /resources/data

# Open classified tweets file and split by line into a list
classifiedTweetsList = open('classified_tweets.txt','r').read().split('\n')
del classifiedTweetsList[-1] # Delete the last row since it's blank

# Initialize arrays to store tweets and true score
num_tweets = len(classifiedTweetsList)
classified_tw = num_tweets * [0]
truths = np.zeros(num_tweets)

# Loop through each line of the file and populate truth and tweet arrays
i = 0  # Iterator
for tw in classifiedTweetsList:
    # First index of each line in file contained the true score
    # Typecast as int and divide by 4 to get 0 for negative, and 1 for positive
    truths[i] = int(tw[0]) / 4

    classified_tw[i] = tw[2:]  # Store the tweet part of the line in the classifier
    i += 1  # increment iterator

# Create dataframe with classified tweet and true score
classifiedDF = pd.DataFrame(
    {'tweet':classified_tw,
     'truth':truths}
)
print(classifiedDF.head())

# Split the dataframe into data and labels
X = classifiedDF['tweet']
y = classifiedDF['truth']

# Initialize vectorizer
# This vectorizer will be used to get unique features from the tweets
vectorizer = TfidfVectorizer(stop_words='english') # Start vectorizer

### 1b) Creating models ###

# The purpose of this section is to find the optimal algorithm. To do so, a greedy
# search technique is used to built models with various parameters. The test parameters
# were chosen randomly from some test runs. The built model is then validated using a
# K-Fold Cross Validation technique since this allows for the testing of a model with
# less variance than a single train-test set split.

# Initialize results dataframe and testing parameters
parameter_list = [0.0001,0.001,0.01,0.1,1,5,10]
index = ['model','param value','mean','std dev']
resultsDF = pd.DataFrame(data=np.random.randn(len(parameter_list)*3,4),columns=index)

# Grid searh w/ K-Fold Cross Validation
all_features = vectorizer.fit_transform(X)
num_folds = 10
i = 0
print('\n')
for param in parameter_list:
    Kfold = KFold(len(y), n_folds=num_folds)

    multiNBModel = MultinomialNB(alpha=param)
    results = cross_val_score(multiNBModel, all_features, y, cv=Kfold)
    print(
    "Multinomial NB Model w/ C=%.3f: Accuracy: %.2f%% (%.2f%%)" % (param, results.mean() * 100, results.std() * 100))
    resultsDF.iloc[i] = ("MultiNB", param, results.mean(), results.std())
    i += 1

    bernNBModel = BernoulliNB(alpha=param)
    results = cross_val_score(bernNBModel, all_features, y, cv=Kfold)
    print(
    "Bernoulli NB Model w/ C=%.3f: Accuracy: %.2f%% (%.2f%%)" % (param, results.mean() * 100, results.std() * 100))
    resultsDF.iloc[i] = ("BernNB", param, results.mean(), results.std())
    i += 1

    logModel = LogisticRegression(C=param)
    results = cross_val_score(logModel, all_features, y, cv=Kfold)
    print("Log Model w/ C=%.3f: Accuracy: %.2f%% (%.2f%%)" % (param, results.mean() * 100, results.std() * 100))
    resultsDF.iloc[i] = ("Log", param, results.mean(), results.std())
    i += 1

# Results
print('\n',resultsDF.sort_values(['model','mean']).head())
print('\n',resultsDF.groupby('model').describe()[['mean','std dev']])

### 1c) Analyzing Performance of Models ###

# Plot mean of all algorithms to visually see performance changes with changing parameters
all_params = resultsDF['param value'].unique()

# Get means
log_mean = resultsDF[resultsDF['model'] == 'Log'].sort_values('param value')['mean'].reset_index()['mean']
multiNB_mean = resultsDF[resultsDF['model'] == 'MultiNB'].sort_values('param value')['mean'].reset_index()['mean']
bernNB_mean = resultsDF[resultsDF['model'] == 'BernNB'].sort_values('param value')['mean'].reset_index()['mean']

# Get standard deviations
log_std = resultsDF[resultsDF['model'] == 'Log'].sort_values('param value')['std dev'].reset_index()['std dev']
multiNB_std = resultsDF[resultsDF['model'] == 'MultiNB'].sort_values('param value')['std dev'].reset_index()['std dev']
bernNB_std = resultsDF[resultsDF['model'] == 'BernNB'].sort_values('param value')['std dev'].reset_index()['std dev']

fig, ax = plt.subplots(figsize=(12,5))

ax.errorbar(all_params, log_mean, yerr=log_std, fmt='-o',ecolor='g',color='g') # plot log
ax.errorbar(all_params, multiNB_mean, yerr=multiNB_std, fmt='-o',ecolor='r', color='r') # plot MultiNB
ax.errorbar(all_params, bernNB_mean, yerr=bernNB_std, fmt='-o',ecolor='b', color='b') # plot BernoulliNB

plt.title("Optimal Parameter Analysis")
plt.xlabel('Parameter')
plt.ylabel('Accuracy (%)')
ax.legend(['Logarithmic','Multinomial NB','Bernoulli NB'],loc='center left',bbox_to_anchor=(1.0,0.5))

# For the Logarithmic Regression model, as C increases, the model performance increases
# until c = 3, then it very slowly decreases. Both the Bernoulli and Multinomial Naive
# Bayes model perform poorly at high alpha values. Both are optimal with alpha values
# below or equal to 0.1. The logarithmic model has fairly low standard deviation compared
# to the other two models. The Bernoulli Naive-Bayes model's standard deviation decreases
# as its alpha parameter increases. The Multinomial Naive-Bayes model's standard deviation
# increases as its alpha parameter increases.

# Find the best model for each algorithm type
print(resultsDF.ix[resultsDF.groupby('model')['mean'].idxmax()])

# Get the optimal parameters
bernNB_param = resultsDF.ix[resultsDF[resultsDF['model'] == 'BernNB'][['mean']].idxmax()]['param value'].values[0]
log_param = resultsDF.ix[resultsDF[resultsDF['model'] == 'Log'][['mean']].idxmax()]['param value'].values[0]
multiNB_param = resultsDF.ix[resultsDF[resultsDF['model'] == 'MultiNB'][['mean']].idxmax()]['param value'].values[0]

# Get the model stats
models = resultsDF.ix[resultsDF.groupby('model')['mean'].idxmax()]['model'].reset_index()['model']
means = resultsDF.groupby('model')['mean'].max().reset_index()['mean']
std_devs = resultsDF.ix[resultsDF.groupby('model')['mean'].idxmax()]['std dev'].reset_index()['std dev']

# Create errorbar plot with model statistics
fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.errorbar(np.arange(3), means*100, std_devs*100, fmt='ok', color='r', lw=2,markersize=10)
plt.title("Model Performance Analysis")
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.xlim(-1,3)
plt.ylim(60,75)
plt.xticks(np.arange(3),['Bernoulli Naive Bayes','Logarithmic Regression','Multinomial Naive-Bayes'])

# From the errorplot seen above, it is seen that the Logarithmic model has the best mean
# performance and the lowest standard deviation. The max of the Bernoulli Model is greater
# than the Log model but its standard deviation is much worse than the Logarithmic model.
# This means that its performance can vary vastly. The optimal algorithm in this case is
# the Logarithmic model.

### 1d) Machine Learning Performance Metrics ###

# Get standard machine learning metrics on the optimal machine learning models

# Create training and test sets

# Use the common convention of 33% for testing and 67% for training. Use the vectorizer
# to transform the tweets to Tfidf features.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
feature_train = vectorizer.fit_transform(X_train)
feature_test = vectorizer.transform(X_test)

# Train and evaluate Logarithmic Regression model
print('\n')
logModel = LogisticRegression(C=log_param)
logModel.fit(feature_train,y_train)
logPredictions = logModel.predict(feature_test)

print("Log Model Classification Report")
print(classification_report(y_test, logPredictions))
print("Log Model Confusion Matrix")
print(confusion_matrix(y_test,logPredictions))

TN = confusion_matrix(y_test, logPredictions)[0][0]
FP = confusion_matrix(y_test, logPredictions)[0][1]
FN = confusion_matrix(y_test, logPredictions)[1][0]
TP = confusion_matrix(y_test, logPredictions)[1][1]
total = TN + FP + FN + TP
# Determine accuracy and misclassification rate
accuracy = (TP + TN) / total * 100
misclassification_rate = (FP + FN) / total * 100
print("\n")
print("Accuracy:",accuracy, '%')
print("Misclassification Rate:",misclassification_rate,'%')
print('\n')
print('MAE:', metrics.mean_absolute_error(y_test, logPredictions)*100, '%')
print('MSE:', metrics.mean_squared_error(y_test, logPredictions)*100, '%')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, logPredictions))*100, '%')

# Train and evaluate the Multinomial Naive Bayes model
print('\n')
multiNBModel = MultinomialNB(alpha=multiNB_param)
multiNBModel.fit(feature_train,y_train)
multiNBPredictions = multiNBModel.predict(feature_test)

print("Multinomial Naive Bayes Model Classification Report")
print(classification_report(y_test, multiNBPredictions))
print("Multinomial Naive Bayes Model Confusion Matrix")
print(confusion_matrix(y_test,multiNBPredictions))

TN = confusion_matrix(y_test, multiNBPredictions)[0][0]
FP = confusion_matrix(y_test, multiNBPredictions)[0][1]
FN = confusion_matrix(y_test, multiNBPredictions)[1][0]
TP = confusion_matrix(y_test, multiNBPredictions)[1][1]
total = TN + FP + FN + TP
# Determine accuracy and misclassification rate
accuracy = (TP + TN) / total * 100
misclassification_rate = (FP + FN) / total * 100
print("\n")
print("Accuracy:",accuracy, '%')
print("Misclassification Rate:",misclassification_rate,'%')
print('\n')
print('MAE:', metrics.mean_absolute_error(y_test, multiNBPredictions)*100, '%')
print('MSE:', metrics.mean_squared_error(y_test, multiNBPredictions)*100, '%')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, multiNBPredictions))*100, '%')

# Train and evaluate the Bernoulli Naive Bayes model
print('\n')
bernNBModel = BernoulliNB(alpha=bernNB_param)
bernNBModel.fit(feature_train,y_train)
bernNBPredictions = bernNBModel.predict(feature_test)

print("Bernoulli Naive Bayes Model Classification Report")
print(classification_report(y_test, bernNBPredictions))
print("Bernoulli Naive Bayes Model Confusion Matrix")
print(confusion_matrix(y_test,bernNBPredictions))

TN = confusion_matrix(y_test, bernNBPredictions)[0][0]
FP = confusion_matrix(y_test, bernNBPredictions)[0][1]
FN = confusion_matrix(y_test, bernNBPredictions)[1][0]
TP = confusion_matrix(y_test, bernNBPredictions)[1][1]
total = TN + FP + FN + TP
# Determine accuracy and misclassification rate
accuracy = (TP + TN) / total * 100
misclassification_rate = (FP + FN) / total * 100
print("\n")
print("Accuracy:",accuracy, '%')
print("Misclassification Rate:",misclassification_rate,'%')
print('\n')
print('MAE:', metrics.mean_absolute_error(y_test, bernNBPredictions)*100, '%')
print('MSE:', metrics.mean_squared_error(y_test, bernNBPredictions)*100, '%')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, bernNBPredictions))*100, '%')

# From the above analysis, it is seen that the Logarithmic Regression model had the least
# error of the three models regardless of the method used to calculate error. It also had
# the highest accuracy of the three models for the given test set.

### From the above analysis, it was determined that the best algorithm for this analysis is
### the Logarithmic Regression Classification algorithm. It will be used from now on for the
### second part of the assignment


## Part 2: Categorize Unclassified Tweets by Political Party ##


# Read in unclassified data
unclassifiedTweetsList = open('unclassified_tweets.txt', 'r',encoding='UTF8').read().split('\n')
del unclassifiedTweetsList[-1] # Delete the last row since it's blank

### 2a) Create Sets of Party Related Words ###

# The following words were found by searching through Twitter pages. Firstly, the party
# name was searched for on Twitter, and then related hashtags were found in the results
# and added to the relevant set. Then those hashtags were search for, and its related
# hashtags were added to the relevant set. This process was continued until no more
# specific hashtags could be found.

NDP = set(['ndp','tommulcair','mulcair','ndpldr','ndleg','pj17','thomasmulcair'])
Liberal = set(['liberal','trudeau','justintrudeau','realchange','lpc'])
Conservative = set(['conservative','cpcldr','cpc','m103','stevenharper','harper','tories','ronaambrose','ambrose'])

### 2b) Function to Assign Party To Tweets ###
def getParty(tw_unigram):
    # Start some counters
    num_ndp_words = 0
    num_lib_words = 0
    num_cons_words = 0

    # Loop through unigrams
    for word in tw_unigram:
        if word.lower() in Liberal:
            num_lib_words += 1
        elif word.lower() in NDP:
            num_ndp_words += 1
        elif word.lower() in Conservative:
            num_cons_words += 1

    party = ''

    if (num_lib_words > num_cons_words and num_lib_words > num_ndp_words):
        party = 'Liberal'
    elif (num_cons_words > num_lib_words and num_cons_words > num_ndp_words):
        party = 'Conservative'
    elif (num_ndp_words > num_lib_words and num_ndp_words > num_cons_words):
        party = 'NDP'
    else:
        party = 'Unknown'

    return party

### 2c) Categorize Tweets By Party ###

# Initialize analyzer to tokenize tweets
analyzer = vectorizer.build_analyzer()

# Initialize list to store parties
party = [0] * len(unclassifiedTweetsList)

# Categorize tweets by party
for i in range(len(unclassifiedTweetsList)):
    party[i] = getParty(analyzer(unclassifiedTweetsList[i]))

## Part 3: Predict Scores for Unclassified Tweets ##

# Transform tweets into features using TFidfVectorizer
features_unclassified = vectorizer.transform(unclassifiedTweetsList)
score = logModel.predict(features_unclassified)

# Write into dataframe
unclassifiedDF = pd.DataFrame(
    {'tweet':unclassifiedTweetsList,
     'party':party,
     'score':score
    }
)
print(unclassifiedDF.head())

# Remove unknown tweets

# These tweets could not be categorized so they will be ignored for analysis

unclassifiedDF = unclassifiedDF[unclassifiedDF['party'] != 'Unknown'].reset_index()[['party','score','tweet']]
print(unclassifiedDF.head())

## Part 4: Political Insights ##

# Overall standard statistical analysis
print(unclassifiedDF.groupby('party').describe())

# Write important stats into dataframe
party_stats = unclassifiedDF.groupby('party').mean()
party_stats['std'] = unclassifiedDF.groupby('party').std()
party_stats['count'] = unclassifiedDF.groupby('party').count()['score']
party_stats['mean'] = party_stats['score']
party_stats['tweet %'] = party_stats['count']/party_stats['count'].sum()*100
party_stats = party_stats.reset_index().drop(labels='score',axis=1)
print('\n',party_stats)

# Analyze each party's buzz
print('\n')
print("The %s party had the most buzz with about %.2f%% of the party related tweets."
      % (party_stats.ix[party_stats['tweet %'].idxmax()]['party'],
         party_stats.ix[party_stats['tweet %'].idxmax()]['tweet %']))

print("The %s party had the least buzz with about %.2f%% of the party related tweets."
      % (party_stats.ix[party_stats['tweet %'].idxmin()]['party'],
         party_stats.ix[party_stats['tweet %'].idxmin()]['tweet %']))

# Analyze which party is the most and least liked
print('\n')
print("The %s party had the most liked with a postive-negative tweet ratio of %.2f%%."
      % (party_stats.ix[party_stats['mean'].idxmax()]['party'],
         party_stats.ix[party_stats['mean'].idxmax()]['mean']*100))

print("The %s party had the least liked with a postive-negative tweet ratio of %.2f%%."
      % (party_stats.ix[party_stats['mean'].idxmin()]['party'],
         party_stats.ix[party_stats['mean'].idxmin()]['mean']*100))

# Plot distribution of sentiments for each party
fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.errorbar(np.arange(3), party_stats['mean'], party_stats['std'], fmt='ok', color='r', lw=2,markersize=10)
plt.title("Party Sentiment Analysis")
plt.xlabel('Party')
plt.ylabel('Score')
plt.xlim(-1,3)
plt.ylim(0.2,1.25)
plt.xticks(np.arange(3),party_stats['party'])
plt.show()

# The Liberal Party performed the best with the highest mean score and lowest standard
# deviation. The NDP party was close behind. The Conservative party performed the worst
# with the lowest mean score and the highest standard deviation.

## Part 5: Conclusion ##

# Conclusion from above results

# The  Liberal and NDP parties have very similar distributions but the Liberal party
# was more liked and had almost twice as many tweets than the NDP party. Overall, it
# shows the Liberal party had the most positive attention and the most likely to win
# the Canadian Federal Election. The Conservative party also got more positive attention
# than the NDP party even though its positive-negative ratio was worse than the NDP party.
# The NDP party had the least attention overall.

# Actual results from the 2015 Canadian election

# This was reflected in the actual 2015 Canadian Federal election because Justin Trudeau
# and the Liberal party won the election and had the most seats. The Conservative party
# and Steven Harper got second most votes with the second most seats. They however lost
# the most seats from what they had prior. The NDP was third with the least votes and
# least seats.

# Ref: https://en.wikipedia.org/wiki/Results_of_the_Canadian_federal_election,_2015