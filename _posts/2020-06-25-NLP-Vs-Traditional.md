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

<img src="https://i.imgur.com/B5Ga3GZ.png" alt="drawing" width="400" height="300"/>

My target is not to have a 95% accurate model, it is to detect *fraud postings* with 70% recall, while maintaining above 95% recall on non fraud postings. 

Is that goal too optimistic? Probably, but I'm bored so lets try it. I figured a business who's hosting these posts would like to get rid of over half of frauds while maintaining a near perfect score with taking down legitimate posts by accident. This model will allow anyone to copy and paste a job posting and get returned whether or not it is likely fraudulent or not.

My problem is a classification problem. Based on this data I need to create a binary classification model that can predict if a job posting is fraudulent (1) or not (0). As a baseline I will be considering the predicting power, in particular the **recall score** of traditional machine learning methods while NOT using any text processing other than ordinal encoding and other categorizing methods. Leakage is not an issue to consider with this data set (as long as I don't get carried away with some feature engineering) as no data other than the "is Fraudulent" column is a dead give away.

**Welp, let's begin. Traditional Machine learning vs. Natural Language Processing**

# Traditional Machine Learning Algorithm
### Tldr; It lost...Bad

I want to really give the traditional machine learning algorithm a chance so I've done some feature engineering to give it some umpf.
