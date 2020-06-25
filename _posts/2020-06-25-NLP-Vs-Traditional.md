---
layout: post
title: I Compare Natural Language Processing Vs. Traditional Machine Learning Models
subtitle: Let's see if doing NLP is really neccessary when predicting fraud in job posts
cover-img: /assets/img/path.jpg
tags: [NLP, Text Processing, Machine Learning]
---

To introduce my little escapade, I'll be looking at a data set that looks at about 20,000 job postings and tells you which ones are actually fraud postings. A link to this data set as well as my full Jupyter Notebook can be found at the bottom if you so wish to investigate my findings further.

## The Preliminaries

Before I begin diving in I want to establish some evaluation metrics and baselines so you can follow exactly what I'm trying to do. I will be evaluating based on a recall score instead of accuracy score. This is because there is a **massive imbalance between the amount of fraud postings and Non-fraud postings. There are 17014 non-fraud yet only 866 fraud postings, which gives me an imbalance of 95%**
