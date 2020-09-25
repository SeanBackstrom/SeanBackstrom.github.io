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
  
# Creating the Naive Bayes Classifier

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


Now I need to be able to calculate probability. I decided to use a gaussian probability density function which assumes x values are drawn from a distribution such as a bell curve. Any non-negative function which integrates to 1 (unit total area) is suitable for use as a probability density function. The most general Gaussian PDF formula is given (here)[https://ccrma.stanford.edu/~jos/sasp/Gaussian_Probability_Density_Function.html]

      def calculate_probability(x, mean, std):
        '''calculates gaussian probability density using f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
        returns the probability of x'''

        exponent = exp(-((x-mean)**2 / (3 * std**2)))
        return (1 / (sqrt(2 * pi) * std)) * exponent
        
I've added a fit function which applies some of the functions to the data, and can be used just as any other .fit method on a machine learning algorithm.

    def fit(dataset):
        '''splits dataset by class and calculates stats for each row'''

        seperated = seperate_by_class(dataset)
        summaries = dict()
        for class_value, rows in seperated.items():
            summaries[class_value] = summarize_dataset(rows)
        return summaries
        
        
furthermore, I need to now calculate probability for a class, not just an input number. each prediction should have a probability for all possible classes. for example a return might be (0.75, 0.20, 0.05) if there was 3 possible classes. According to Bayes Theorem, the probability that a piece of data belongs to a class is calculated as follows:

P(class|data) = P(X|class) * P(class)

The input variables are treated separately, giving the technique it’s name “naive“. For the above example where we have 2 input variables, the calculation of the probability that a row belongs to the first class 0 can be calculated as:

`P(class=0|X1,X2) = P(X1|class=0) P(X2|class=0) P(class=0)`

assuming this prints a most likely probability, this is a complete bayes algorithm. I coded the algorithm as follows:

    def calculate_class_probability(summaries, row):
        '''calculate the probabilities of predicting each class for a given row. P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0) for each class in the dataset. returns dict of probabilites with one entry for each class'''

        #summaries 0,2 is length of data
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, std, _ = class_summaries[i]
                probabilities[class_value] *= NaiveBackstromClassifier.calculate_probability(row[i], mean, std)
        return probabilities



Finally, I had to create a predict function that works like a typical machine learning predict method. My final Naive Backstrom Classifier looks like this!

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

      def fit(dataset):
          '''splits dataset by class and calculates stats for each row'''

          seperated = seperate_by_class(dataset)
          summaries = dict()
          for class_value, rows in seperated.items():
              summaries[class_value] = summarize_dataset(rows)
          return summaries

      def calculate_probability(x, mean, std):
          '''calculates gaussian probability density using f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
          returns the probability of x'''

          exponent = exp(-((x-mean)**2 / (3 * std**2)))
          return (1 / (sqrt(2 * pi) * std)) * exponent

      def calculate_class_probability(summaries, row):
          '''calculate the probabilities of predicting each class for a given row. P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0) for each class in the dataset. returns dict of probabilites with one entry for each class'''

          #summaries 0,2 is length of data
          total_rows = sum([summaries[label][0][2] for label in summaries])
          probabilities = dict()
          for class_value, class_summaries in summaries.items():
              probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
              for i in range(len(class_summaries)):
                  mean, std, _ = class_summaries[i]
                  probabilities[class_value] *= NaiveBackstromClassifier.calculate_probability(row[i], mean, std)
          return probabilities

      def predict(summaries, row):
          '''predict the class for a given row'''
          probabilities = NaiveBackstromClassifier.calculate_class_probability(summaries, row)
          best_label, best_prob = None, -1
          for class_value, probability in probabilities.items():
              if best_label is None or probability > best_prob:
                  best_prob = probability
                  best_label = class_value
          return best_label
        
In typical data scientist fashion, I decided to test it using the iris dataset. All I have to do is load it up and fit the model on it, and I should be able to predict a new row of data (that I will create on the spot) and the probability of what class it is in:

    #df is iris.csv
    model = NaiveBackstromClassifier.fit(df)
    #fake data to predict class
    row = [5.7,2.9,4.2,1.3]
    label = NaiveBackstromClassifier.predict(model, row)
    print('Data=%s, Predicted: %s' % (row, label))

Output:

    Data=[5.7, 2.9, 4.2, 1.3], Predicted: 2
    
Success! It predicted that it is in the second class, which is 'Iris-setosa'
