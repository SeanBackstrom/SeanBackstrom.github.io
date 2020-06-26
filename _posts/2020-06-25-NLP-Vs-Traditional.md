---
layout: post
title: I Compare Natural Language Processing Vs. Traditional Machine Learning Models
subtitle: Let's see if doing NLP is really neccessary when predicting fraud in job posts
cover-img: /assets/img/path.jpg
tags: [NLP, Text Processing, Machine Learning]
---

To introduce my little escapade, I'll be looking at a data set that looks at about 18,000 job postings and tells you which ones are actually fraud postings. I will be creating a machine learning algorithm to be able to predict what postings are fraud. A link to this data set as well as my full Jupyter Notebook can be found at the bottom if you so wish to investigate my findings further.

## The Preliminaries

I will be evaluating based on a recall score instead of accuracy score.([For info on difference between recall and accuracy](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c))  This is because **there is a massive imbalance between the amount of fraud postings and non-fraud postings.** There are 17014 non-fraud postings and only 866 fraud postings, which gives me a class imbalance of 95%. 

<img src="https://i.imgur.com/g0zhpJe.png" alt="Imbalance graph" width="450" height="300"/>

My target is not to have a 95% accurate model, it is to detect *fraud postings* with 70% recall, while maintaining above 95% recall on non fraud postings. 

Is that goal too optimistic? Probably, but I'm bored so lets try it. I figured a business who's hosting these posts would like to get rid of over half of frauds while maintaining a near perfect score with taking down legitimate posts by accident. This model will allow anyone to copy and paste a job posting and get returned whether or not it is likely fraudulent or not.

My problem is a classification problem. Based on this data I need to create a binary classification model that can predict if a job posting is fraudulent (1) or not (0). As a baseline I will be considering the predicting power, in particular the **recall score** of traditional machine learning methods while NOT using any text processing other than ordinal encoding and other categorizing methods. Leakage is not an issue to consider with this data set (as long as I don't get carried away with some feature engineering) as no data other than the "is Fraudulent" column is a dead give away.

**Welp, let's begin. Traditional Machine learning vs. Natural Language Processing**


# Traditional Machine Learning Algorithm

### Tldr; It lost...Bad

I want to really give the traditional machine learning algorithm a chance so I've found lots of features other than the job description to find giveaways of fraud postings. 

First, I've deleted columns such as "Job Description", and "benefits" as they are handwritten style, and can not be categorized. I've kept lots of columns such as "Location" and "department" which are categorical and can be numericalized. 

Second, I noticed fraud postings tend to leave a lot of fields not filled out which puts them as NaN's in my dataset. Here you can compare shapes of how many NaN's are in typical Non Fraud - Fraud postings. 

<img src="https://i.imgur.com/Ra2Tvf3.png" alt="nancount graphs"/>
<p style="text-align: center;"><sub> Bias alert (ʘᗩʘ') : The difference in the scale of Y between graphs can potentially explain the variance away. </sub></p>

Third, like this graph you can find lots of differences between fraud and non fraud job postings when you plot them out without having to do any language processing.

<img src="https://i.imgur.com/zZ7kUWA.png" alt="Experience level graph"/>

## Seems good! Let's try out a model.

### Logistic Regression

I'll be starting with a logistic regression model, and I'll be categorizing many of the text fields.
```python
lr = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(), 
    LogisticRegression(n_jobs=-1)
)
lr.fit(X_train, y_train)
linpredicted = lr.predict(X_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, linpredicted))
print("Logistic Regression Recall:", recall_score(y_val, linpredicted))
```
*drum roll*

**Logistic Regression Accuracy: 0.951**

Wow it did great right?! Wait...

**Logistic Regression Recall: 0.007**

Ouch... That didn't work at all. As you can see it did almost a perfect job predicting nonfraud, and did just about a perfect job **not** predicting fraud.

0 = nonfraud and 1 = fraud

<img src="https://i.imgur.com/S2yFjkh.png"/>

## It's time for the big guns.

### XGBoost Classifier

We haven't evolved science in machine learning to stop at logistic regression, so I'm bringing out the big guns with XGBoost Classifier to hopefully beef up our recall score. After running it with the same parameters above the output is:

**XGBoost Classifier Accuracy: 0.960**

**XGBoost Classifier Recall: 0.237**

and after testing with my final test set we get downgraded to:

**XGBoost Classifier Accuracy: 0.957**

**XGBoost Classifier Recall: 0.150**

<img src="https://i.imgur.com/rLKgk62.png"/>

Oof. Quite a shy piece short of my 70% hoped accuracy, but **we got a nice baseline to start with.** Now let's get to the big boy toys.


# Natural Language Processing

I don't want this to come across as an NLP tutorial so I'll skip most of the depth of converting the text into tokens and lemmatizing and such. ([Here's a great introduction anyways](https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958)) 
