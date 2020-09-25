---
layout: post
title: I Create a homemade Naives Bayes Classifier Machine Learning Algorithm
subtitle: Warning. Math incoming.
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [Algorithms, Machine Learning]
comments: true
---

  While barreling through my data science projects I've been having moments thinking to myself, how on earth is this algorithm really doing this?
  I mean I know the concepts and general theory behind them, or perhaps an analogy that makes sense for when to use what algorithm, but after one trip into the source
  code of a K-Nearest-Neighbors algorithm, I knew I knew just about nothing. 
  Hence, this project was born out of my desire to know a little bit more about what is going on under the hood of the algorithms we are blessed to get to write in with a few lines of code.
  
  
  I created a Naive Bayes Classifier from scratch in Python using just base python, and a little numpy. I picked a Naive Bayes Classifier because I wanted to learn about Bayes theorem while also learning about algorithm creation too! No further adu, let's create.
  
# Creating the Naive BClassifier

Because this is a "class"ifier, first up is to make some functions to split up data by class and to make a simple function that returns a math summary of the dataset (returning the mean, std, and data length for example). I call it NaiveBackstromClassifier because hey, I made it! I can name it after myself.

    class NaiveBackstromClassifier():
    # Functions neccesary to streamline NBC

    
      def summarize_dataset(dataset):
          '''returns important math functions for data'''

          summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
          del(summaries[-1])
          return summaries

      def seperate_by_class(dataset):
          '''split data by class values, returns in dict. assumes last column in dataset is class value'''

          seperated = dict()
          for i in range(len(dataset)):
              vector = dataset[i]
              class_value = vector[-1]
              if (class_value not in seperated):
                  seperated[class_value] = list()
              seperated[class_value].append(vector)
          return seperated

need access to std, mean, total data amounts a lot
need to calculate probability. To follow NaiveBayes, use gaussian probability density

assumes x values are drawn from a distribution such as a bell curve
Any non-negative function which integrates to 1 (unit total area) is suitable for use as a probability density function. The most general Gaussian PDF is given by shifts of the normalized Gaussian: formula is here: https://ccrma.stanford.edu/~jos/sasp/Gaussian_Probability_Density_Function.html
need to now calculate probability for a class, not just a input number. each prediction should have a probability for all possible classes. for example a return might be (0.75, 0.20, 0.05)

The probability that a piece of data belongs to a class is calculated as follows:

P(class|data) = P(X|class) * P(class)

The input variables are treated separately, giving the technique it’s name “naive“. For the above example where we have 2 input variables, the calculation of the probability that a row belongs to the first class 0 can be calculated as:

P(class=0|X1,X2) = P(X1|class=0) P(X2|class=0) P(class=0)

assuming this prints a most likely probability, this is a complete bayes algorithm

create predict function
manages the calculation of the probabilities of a new row belonging to each class, and select the class with largest prob value
