# [Contagious Naive Bayes(CNB)](https://github.com/iEna101/Contagious-Naive-Bayes)

With the increase in online social media interactions, the true identity of user profiles becomes increasingly doubtful. 
Fake profiles are used to engineer perceptions or opinions as well as to create online relationships under false pretence. 
Natural language text – how the user structures a sentence and uses words – provides useful information to discover expected patterns, 
given the assumed social profile of the user. We expect, for example, different word use and sentence structures from teenagers to those of adults. 

Sociolinguistics is the study of language in the context of social factors such as age, culture and common interest. Natural language processing (NLP),
provides quantitative methods to discover sociolinguistic patterns in text data. Current NLP methods make use of a multinomial naive Bayes classifier to 
classify unseen documents into predefined sociolinguistic classes. One property of language that is not captured in binomial or multinomialmodels, 
is that of burstiness. Burstiness defines the phenomenon that if a person uses a word, they are more likely to use that word again. 
Thus, the independence assumption between respective counts of the same word is relaxed. The Poisson distribution family captures this phenomenon and 
in the field of biostatistics, it is often referred to as contagious distributions (because the counts between contagious diseases are not independent). 
In this pacakge, the count independence assumption of the naive Bayes classifier is relaxed by replacing the baseline multinomial likelihood function with 
a Poisson likelihood function. 

This packages thus allows the user to make use of contagious naive Bayes as an alternative to the readily available techniques to perform binary text classification.  

It is important to note that when making use of the package, the package itself contains built-in preprocessing. The details of the preprocessing performed can be found in the following [script](https://github.com/iEna101/Contagious-Naive-Bayes/blob/master/Preprocessing.ipynb).

## Getting Started:

The package is available [online](https://pypi.org/project/Contagious-Naive-Bayes/1.0.2/) for use within Python 3 enviroments.

The installation can be performed through the use of a standard 'pip' install command, as provided below: 

`pip install Contagious-Naive-Bayes==1.0.2`

## Prerequisites:

The package has several dependencies, namely: 

* pandas
* numpy
* re
* nltk
* warnings
* sklearn
* BeautifulSoup

## Function description:

The function is named **CNB**.

The function can take 6 possible arguments, two of which are required, and the remaining 4 being optional. 

### The required arguments are: 

-**Train_Matrix**(A matrix containing the observations on which the classification model will be trained.)

-**Test_Matrix**(A matrix containing observation which will be used to test the performance of the model.)

### The optional requirements are: 

-**norm**(A True/False flag which specifies whether document length normalization must be applied. The method of document length normalization utilized for this package this that of Pseduo-Length normalization. *The default is set to False.*)

-**pseudo_len**(Should document length normalization be required, this specifies the length to which the documents should be normalized to. *The default is 100, while any user input is 
required to be postive.*)

-**c1**(This is the first smoothing parameter required to perform document length normalization. *The default is set to 0.001.*)

-**c2**(This is the second smoothing parameter required to perform document length normalization. *The default is set to 1.*)

## Output:

The function provides the output in two components, firstly it provides a table containing the index of the observations, the posteriors calculated per possible class for each observations as well as the predicted class and the actual class of each observation. 

Secondly, the function provides several performance metrics, the metrics utilized are accuracy, precision, recall and the F1 score. 

## Example Usage:

A more comprehensive [tutorial](https://github.com/iEna101/Contagious-Naive-Bayes/blob/master/Tutorial.ipynb) is also available.  

### Installation;

Run the following command within a Python command window:

`pip install Contagious-Naive-Bayes==1.0.2`


### Implementation;

Import the package into the relevant python script, with the following: 

`from Contagious_NB import Classification as func`

> Call the function:

#### Possible examples of calling the function are as follows:

`x_cnb = func.CNB(train_matrix,test_matrix)`

`x_cnb = func.CNB(train_matrix,test_matrix, norm = True,  pseduo_len = 100, c1 = 0.001, c2 = 1)`


### Results;

The output obtained appears as follows: 

![Post](/Images/Post.png)


![Metrics](/Images/Metrics.png)

## Built With:

[Google Collab](https://colab.research.google.com/notebooks/intro.ipynb) - Web framework

[Python](https://www.python.org/) - Programming language of choice

[Pypi](https://pypi.org/) - Distribution

## Authors:

[Iena Petronella Derks](https://github.com/iEna101/Contagious-Naive-Bayes)


## Co-Authors:

Alta de Waal

Luandrie Potgieter

[Ricardo Marques](https://github.com/RicSalgado)


## License:

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments:

University of Pretoria 
![Tuks Logo](/Images/UPlogohighres.jpg)

Center for Artifcial Intelligence Research (CAIR)
![CAIR Logo](/Images/cair_logo.png)
