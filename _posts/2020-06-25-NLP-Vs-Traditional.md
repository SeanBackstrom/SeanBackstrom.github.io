---
layout: post
title: Too good to be true? I write an effective predictive model on whether a job posting is fake or real.
subtitle: Comparing Natural Language Processing Vs. Traditional Machine Learning Models
cover-img: /assets/img/path.jpg
tags: [NLP, Text Processing, Machine Learning]
---
# Tired of these?

<img src="https://i.imgur.com/1Dclctx.png"/>

Me too. So I've designed an algorithm that can be used to filter them out.

To introduce my little escapade, I'll be looking at a data set that looks at about 18,000 job postings and tells you which ones are actually fraud postings. I will be creating a machine learning algorithm to be able to predict what postings are fraud. A link to this data set as well as my full Jupyter Notebook can be found at the bottom if you so wish to investigate my findings further.

## The Preliminaries

I will be evaluating based on a recall score instead of accuracy score. ([For info on difference between recall and accuracy](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c))  This is because **there is a massive imbalance between the amount of fraud postings and non-fraud postings.** There are 17014 non-fraud postings and only 866 fraud postings, which gives me a class imbalance of 95%. 

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

<img src="https://i.imgur.com/S2yFjkh.png?1"/>

## It's time for the big guns.

### XGBoost Classifier

We haven't evolved science in machine learning to stop at logistic regression, so I'm bringing out the big guns with XGBoost Classifier to hopefully beef up our recall score. After running it with the same parameters above the output is:

**XGBoost Classifier Accuracy: 0.960**

**XGBoost Classifier Recall: 0.237**

and after testing with my final test set we get downgraded to:

**XGBoost Classifier Accuracy: 0.957**

**XGBoost Classifier Recall: 0.150**

<img src="https://i.imgur.com/rLKgk62.png?1"/>

Oof. Quite a shy piece short of my 70% hoped accuracy, but **we got a nice baseline to start with.** Now let's get to the big boy toys.


# Natural Language Processing

I don't want this to come across as an NLP tutorial so I'll skip most of the depth of converting the text into tokens and lemmatizing and such. ([Here's a great introduction anyways](https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958)) 

## Preparing the text

The jist of what I did was to combine all the text based columns such as job description, location, benefits, requirements, all into one column you might call my big wall of words. I then tokenize them, remove as much punctuation as I can, take out the stop words, Then finish with some lemmatizing. Finally as some seasoning out of the oven I lowercase them all and seperate neatly. Finally I taste test with my typical heavy slang test to make sure it's working nicely. Finally some chees- wait what was I saying? Im hungry...

## Result:

```python
spacy_tokenizer("what? Hold on a second. I thought you were good bro..")
output:
['hold', 'second', 'thought', 'good', 'bro', '..']
```

I want to get my head around what kind of word differences there are so I made a not so scientific wordlcoud just to see the different kinds of words (if there are any) a fraud post might use compared to a nonfraud post.


<img src="https://i.imgur.com/PachyD5.png"/>

As we can see, there is a lot of similiarities but we are seeing some different vocabulary which gives me hope in the mythical recall score >70%.

## The Final Countdown: Fitting the models

To clarify again, This second test, using NLP, is literally just one column of an index, and one column of all the text put together. The taditional method above used over 15 columns to make its (trash quality) decision with all kinds of parameters. The final step before I began is to create a vectorizer using SKLearn's CountVectorizer that I can call upon in my new NLP pipeline.

If you are interested I try a few different model fits including XGBoost but to save you time I only posted the most succesful model I got which was a simple Logistic Regression below.

## Logistic Regression Model

Let's fit this baby.

```python
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', lrm)])

# fitting our model.
pipe.fit(X_train,y_train)

linpredicted = pipe.predict(X_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, linpredicted))
print("Logistic Regression Recall:", recall_score(y_val, linpredicted))
```

Ta da!

**Logistic Regression Accuracy: 0.983**

**Logistic Regression Recall: 0.762**

Now that is an improvement. Here is a confusion matrix with the results.

0 = nonfraud and 1 = fraud

<img src="https://i.imgur.com/EDW3IFt.png"/>

If you aren't familiar with confusion matrix graphs like above, that is saying that 2710 nonfraud cases were correctly predicted, and 12 incorrect. and that 34 fraud jobs were incorrectly predicted and **105 jobs were correctly predicted as fraud.** That's a wrap boys, it worked like a charm. 

Below I've got the most important words discovered to decide whether a job is fraud or not. The more red it is the more it contributes towards being fraudulent: 

<img src="https://i.imgur.com/aJPpNJR.png"/>


While a lot of jibberish is in there forsure, it gives us some nice words to be careful of. (Look at #10 web devs!) 

# Final Test Result

As always, the moment of all truth; the final test results ran on my final test set:

**Logistic Regression Test Accuracy: 0.9840604026845637**

**Logistic Regression Test Recall: 0.7225433526011561**

I was able to succesfully achieve my goal of getting a recall greater than 70% for fraud posts, and had the bonus of getting around 99% accuracy with real postings. That's a wrap from me. Below you can check out my notebook and my source for the dataset.

### Bonus: XGBoostClassifier results from NLP pipeline (beating baseline)

**XGboost Classifier Accuracy: 0.9723872771758126**
**XGboost Classifier Recall: 0.43884892086330934**

# Sources & Work

[Personal Notebook](https://github.com/SeanBackstrom/Unit2Build/blob/master/unit-2-build.ipynb) 

[Source of dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction) 
