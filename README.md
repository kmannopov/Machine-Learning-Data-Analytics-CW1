# Coursework for Module 6COSC017C-n - Machine Learning and Data Analytics

Learning Objectives and Outcomes:
- Given a large data set, implement data mining/machine learning techniques focused on problem analysis, data pre-processing, data post-processing.
- Choose and implement appropriate algorithms.
- Perform critical evaluation of the performance of data mining and machine learning algorithms for a given domain/application.
- Produce a written report on the analysis of the data set

# Introduction

Our dataset deals with clients of a certain bank. The set contains
10,000 entries, each of which has a parameter for whether they are
employed, their bank account balance, their annual salary, and whether
they have defaulted on their loans. According to the dataset source on
Kaggle, the data is based on real clients from a bank, just without any
personally identifiable information. The purpose of our project is to
build a machine learning model which can reliably and precisely predict
whether a client will default on their loan, based on their other
features. We will then compare it to two other models, and see which one
is optimal for our purpose. Most, if not all banks already use their
custom algorithms and models to calculate the risk for a loan, except
they have access to a more extensive credit history of the client, which
can improve the accuracy of their calculation.

# Exploratory Data Analysis

The dataset initially has 5 columns: Index, Employed, Bank Balance,
Annual Salary, and Defaulted. Index is simply the number of the
row/entry, so it can be ignored. Two of our variables (Employed and
Defaulted) are categorical, while the other two (Bank Balance and Annual
Salary) are numerical. Our target for the prediction will be the
Defaulted variable, as we are trying to predict whether a certain client
will default on their loan. That leaves us with three features.

Measures of Central Tendency:

![image](https://user-images.githubusercontent.com/79659647/175014282-0c6bda6b-c606-413e-aec2-9872c4b37929.png)

Variance:

Employed 2.077494e-01

Bank Balance 3.369314e+07

Annual Salary 2.561270e+10

Defaulted 3.219433e-02

The shape of our dataset is two
dimensional, with 10,000 rows and 5 columns. The average bank balance is
10,024, while the average annual salary is 402,203. We can see the
distribution for both variables on the following graphs:
![image](https://user-images.githubusercontent.com/79659647/175014447-fe7164fe-27e7-4cfe-a432-c7a7b94e88c5.png)
![image](https://user-images.githubusercontent.com/79659647/175014463-833ee2dc-8f43-4867-accb-95efdac8ab1e.png)

As we can see, the distribution of annual salary for clients that *have
not* defaulted somewhat resembles a standard bell curve, while the
distribution for clients that *have* defaulted is slightly skewed to the
left, but nothing significant.

The distribution of bank balance for non-defaulted clients is heavily
skewed to the left, while the opposite is true for defaulted clients.

By looking at the scatterplot, we can see that there is a correlation
between a higher bank balance and defaults:
![image](https://user-images.githubusercontent.com/79659647/175014522-64620284-75ea-4d13-80ae-d8fd9fc1c1ff.png)

This observation is reinforced by our correlation matrix:

![image](https://user-images.githubusercontent.com/79659647/175014557-31e14762-c8d1-4b43-83fc-2affc6f21623.png)

There is also a high correlation between annual salary and employment,
which isn't much of a surprise. What is interesting, however, is the
lack of correlation between annual salary and bank balance. One might
assume that a higher salary would lead to a higher balance, but other
factors, such as financial responsibility, or the possibility of the
client having other bank accounts are likely at play.

![image](https://user-images.githubusercontent.com/79659647/175014592-657f128b-c9d5-4f3d-8645-221f5b1bb6ce.png)

Our annual salary boxplots are similar for both defaulted and
non-defaulted clients, except the range for defaulted clients is
smaller, with a higher minimum and lower maximum than non-defaulted
clients.

# Data Preparation

To prepare our dataset, the first step was to convert our Booleans into
integers. We copied our dataset into a separate variable, and used the
astype() function to convert the necessary columns.

Next, we added a new feature called 'Balance over Salary', which takes
the Bank Balance and divides it by their annual salary. Since we have a
very limited number of features to begin with, this seemed like a
potentially useful variable, of which we can later test and check the
importance. We also dropped our 'Index' column at this stage, because
the numeration of each entry is an unnecessary variable which can
potentially affect the outcome if the model takes it into account.

In addition, we checked for null values using .isnull().sum(), but we
didn't have any null values.

Before splitting the data, we scaled it using min-max normalization,
which converts the data into values between 0 and 1. In addition, we
have a separate training and testing set that is not scaled, as one of
the models saw worse results when using a scaled dataset.

Finally, we split the training and testing data with a 90-10 split.
Initially, the split was 80-20, but I found that we could improve the
recall score by increasing the size of the training set.

# Model Choice Justification

The metric we will be using to evaluate the performance of our models is
recall. For a bank, minimizing the number of false negatives is crucial,
because the bank does not lose any money if they deny a loan to a paying
client, but they do suffer a loss if they loan money to a client that
will default on the loan. Since recall evaluates the performance of a
model based on false negatives, it is the optimal choice. However, we
can also use precision to see if our model has a high false positive
rate as well, as the bank may not want to lose *too many* potential
borrowers. To summarize, recall will have higher significance, but
precision will play a role as well.

Since the dataset we are working with requires us to predict a state --
whether a client will default on their loans or not, our algorithms will
need to specialize in classification. This means that the models will be
supervised.

Our first model is a Decision Tree model. According to Corporate Finance
Institute (2021), lenders use a decision tree to predict the probability
of a customer defaulting on a loan, which is our exact use case. We can
also compare the results of this model, to see if it is the best model
for our use case. In addition, a decision tree can show the importance
of each of our variables and how much they can affect the default
probability of each client.

Hyper-parameters:

**Criterion=entropy**. By default, scikit uses the gini index to measure
impurity, but we were able to increase our recall metric by changing it
to use entropy.

**Random_state=0**. The metrics of the model seemed to change slightly
each time it was retrained. To eliminate this, we change this parameter
so that the model does not randomize the training data each time.

**Max_depth=8**. By limiting the depth of our decision tree, we were
able to further increase our recall metric. I believe this is due to the
model placing less importance on certain features when we limit the
depth, as less branches mean less decisions to be made. Limiting it
lower than 8 had a negative impact.

Our second model is a Naïve Bayes model. The premise of this model is
that it assumes that variables are completely independent of each other,
and any connection between variables should have no effect on the
outcome. When looking at our scatterplot earlier, we saw that a variety
of clients defaulted on their loans, with annual salary not having an
apparent correlation with 'default' probability. However, we saw that
our dataset had no clients with a bank balance lower than 7,000 that
defaulted. This implies that there may be no link between salary and
bank balance. To specifically test this, we chose a Naïve Bayes model.

Hyper-parameters: none.

Our third model is a Logistic Regression model. Going back to our
scatterplot, we can see that defaulted clients is linearly separable
(along the bank balance of \~7,000). Logistic Regression models perform
well with such kinds of data, as it operates on a linear decision
surface. In addition, a small number of columns ensured that the model
had no risk of over-fitting the data. Furthermore, with a logistic
regression model, we can see the weight of each feature, and compare it
to our findings from the decision tree importance metrics.

Hyper-parameters: none.

# Model Evaluation

  Algorithm   | TPR        | TNR       |  AUC      |   Recall  |    Precision |
  ------------| -----------| ----------|-----------|-----------|--------------|
  Decision Tree  | 0.(45)     | 0.993     |  0.724    |  0.(45)   |   0.68(18)   |
  Naïve Bayes | 0.(3)      | 0.996     |   0.665   | 0.(3)     | 0.7(3)       |
  Logistic Regression   | 0.(3)      | 0.997     |  0.665    |  0.(3)    |   0.786      |

Feature Importance

In the case of both the decision tree and logistic regression models,
Bank Balance seemed to play the most important role in predicting
whether a client will default on their loans. Employment status and
Annual Salary seemed to have some effect but were both within the margin
of error.

# Conclusion

In conclusion, the Decision Tree model ended up being the most fitting
model for our use case. While it does have a lower precision score, the
difference is not enough to offset the benefit of the higher recall
score. We can now see why lenders choose to use this specific model in
this scenario. Even though the recall score is higher than the
alternatives, I would assume that it is nowhere near what is the
industry standard for banks. In fairness, banks have access to a
plethora of information about the customer, as well as an extensive
credit history, which can greatly impact the outcome of the model. To
improve the model in the future, more features need to be added.
Variables such as number of on time payments, total outstanding balance,
and credit scores (such as the ones used in the United States and
Canada) are much more valuable when deciding the creditworthiness of
someone.
