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

![ImbalanceGraph](https://i.imgur.com/GD9YoOv.png)

My target is not to have a 95% accurate model, it is to detect *fraud postings* with 75% accuracy, while maintaining above 95% recall on non fraud postings. Is that too optimistic, probably, but I'm bored so lets try it.
My problem is a classification problem. Based on this data I need to create a binary classification model that can predict if a job posting is fraudulent (1) or not (0).
