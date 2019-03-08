# Project 2: Author Recognition

## Getting Started
`main.py` takes in 4 arguments:
1. Directory of Hamilton texts.
2. Directory of Madison texts.
3. Directory of Disputed texts.
4. List of features as a .txt file.

Example script:
```
python main.py federalist_papers/Hamilton federalist_papers/Madison federalist_papers/Disputed features.txt
```

## Pre-Processing
For preprocessing, we used the nltk library to remove pre-chosen stop words from the texts to be analyzed.
Each word in the filtered text then is stemmed using the nltk.stem.PorterStemmer.PorterStemmer() function.
Finally, each word in the resulting text is lower-cased.

## Trained Features
The features the Naive Bayes model was trained over is located in the `features.txt` file. We used one of the files from the disputed set and excluded it from the testing set.

## Rejected Features / Pre-Processing Techniques
For pre-processing, we decided not to use N-Grams in our project. The results we got were inaccurate when initially using N-Grams due to our program never returning any true string comparisons between the features and the data set.

We did not reject any features.

## Overall Accuracy of Model
When we ran the model with the original amount of data (51 Hamilton texts, 14 Madison texts, 5 Jay texts), the results were skewed towards Hamilton, even when using a majority of features from other authors.  We reduced the amount of Hamilton texts to 20 files and removed Jay entirely. After that, we ended up receiving more accurate representations when testing with our debugging set.

