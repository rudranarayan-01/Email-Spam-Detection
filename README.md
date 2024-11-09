
# Email Spam Detection using Machine Learning

The objective of email spam detection is to accurately classify incoming emails as either legitimate (ham) or spam. 
Traditional rule-based approaches have limited effectiveness due to the constantly evolving nature of spam. Machine 
learning offers a more dynamic and adaptable approach by leveraging patterns and features extracted from large email 
datasets. 




## Mainly two sklearn modules are used to perform this task

### CountVectorizer() :
   Transforms text documents into a matrix that counts the occurrences of tokens (words or n-grams) in each document
    

### MultinomialDB() :
Multinomial Naive Bayes is a probabilistic classifier to calculate the probability distribution of text data.

### So here is two important step i.e Transform the email text and then fit the train data to the model.


### Similarly to check wheather the email is spam or not we need to transform the incoming email into matrix using CountVectorizer and then predict wheather it is spam or ham. 


## Optimizations

To optimize the above two steps we create a sklearn pipeline which will transform the text first and then fit into the model.

###  This model gives 98% of accuracy.


## Installation

Clone my project

```bash
  git clone https://github.com/rudranarayan-01/Email-Spam-Detection-ML-
 
```
Add some feature.. Thank you !!!
## Authors

- [@rudranarayan-01](https://github.com/rudranarayan-01)

