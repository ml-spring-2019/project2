# Project 2: Author Recognition

## Getting Started
`main.py` takes in 5 arguments:
1. Directory of Hamilton texts.
2. Directory of Jay texts.
3. Directory of Madison texts.
4. Directory of Disputed texts.
5. List of features as a .txt file.

Example script:
```
python main.py federalist_papers/Hamilton federalist_papers/Jay federalist_papers/Madison federalist_papers/Disputed features.txt
```

## Preprocessing
We used stop word and stem word filtering to elimate common words from the training and testing datasets.  We also converted all characters to lowercase for streamlined string comparisons.

## Trained Features
The features the Naive Bayes model was trained over is located in the `features.txt` file. 

## Rejected Features / Preprocessing Techniques


## Overall Accuracy of Model
