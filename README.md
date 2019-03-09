# Project 2: Author Recognition

## Getting Started
`main.py` takes in 3 arguments:
1. Directory of Hamilton texts: federalist_papers/Hamilton
2. Directory of Madison texts: federalist_papers/Madison
3. Directory of Disputed texts: federalist_papers/Disputed
The directories have to be in this order: Hamilson, Madison, and Disputed.
Example script:
```
python main.py federalist_papers/Hamilton federalist_papers/Madison federalist_papers/Disputed
```
## Modifying Debugging set
To modify the debugging set, the first option is to manipulate the sub-directories within the existing debugging_data directory so that each of the sub_directories (Hamilton, Madison, then Disputed in order) has the desired text files. The second option is to use a different directory that contains three sub_directories (Hamilton, Madison, then Disputed in order) that respectively contain their own text files.

## Pre-Processing
For preprocessing, we used the nltk library to remove pre-chosen stop words from the texts to be analyzed.
Each word in the filtered text then is stemmed using the nltk.stem.PorterStemmer.PorterStemmer() function.
Finally, each word in the resulting text is lower-cased.

## Trained Features
The features the Naive Bayes model was trained over are 'federalist_papers/Hamilton' and 'federalist_papers/Madison'.
Each directory contain known texts from the respective author.
There are 20 Hamilton text files and 14 Madison text files; we reduced the size of Hamilton text files from 51 to 20 because of simplicity of debugging and quicker runtime.

## Rejected Features / Pre-Processing Techniques
For pre-processing, we decided not to use N-Grams in our project. The results we got were inaccurate when initially using N-Grams due to our program never returning any true string comparisons between the features and the data set. Instead, we implemented the removal of stop-words from both the training and testing texts using the nltk library.

## Overall Accuracy of Model
When we ran the model with the original amount of data (51 Hamilton texts, 14 Madison texts, 5 Jay texts), the results were skewed towards Hamilton, even when using a majority of features from other authors; however, that was because we multiplied P(Author) to each P(word|Author) that existed in the disputed text causing a bias towards the Hamilton text files since there are more Hamilton text files.  We reduced the amount of Hamilton texts to 20 files and removed Jay entirely. Though it was when we edited the Naive Bayes algorithm to have the log of P(Author) only added to the final log of P(disputed text|Author) rather than each log of (word Author) that the classifications of the disputed files became substantially more accurate and the debugging set, correct.
