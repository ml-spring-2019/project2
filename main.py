'''
Technique we may do:
N-Grams

Algorithms to apply when building model:
Laplace smoothing
log-probabilities?

Hamilton: 51 texts
Jay: 5 texts
Madison: 14 texts
Total Known Author Texts: 70 texts
Disputed: 15 texts
'''

import sys
import pdb
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams

# priori probability of each author
hamilton_prob = 51.0/70.0
jay_prob = 5.0/70.0
madison_prob = 14.0/70.0

# need to find the num of all the possible words (from all the texts in the Hamilton, Jay, Madison, and Disputed texts)
possible_words = 1000.0

def main(argv, argc):
    if (argc < 3):
        print("Usage: python main.py <training-set> <testing-set>")
        return 1
    training_file, testing_file = file_IO(argv)

    training_set = get_word_set(training_file)
    testing_set = get_word_set(testing_file)

    bayes_theorem_with_no_divisor(["to"], training_set, jay_prob)

    pdb.set_trace()


def get_word_set(file):
    contents = file_data_to_string(file)
    filtered_stop = remove_stop_words(contents)
    stem = stem_words(filtered_stop)
    lower = lower_case_words(stem)
    return lower

def lower_case_words(words):
    result = []
    for w in words:
        result.append(w.lower())
    return result

def stem_words(words):
    ps = PorterStemmer()
    result = []
    for w in words:
        result.append(ps.stem(w))
    return result


def remove_stop_words(contents):
    words = word_tokenize(contents)
    stop_words = set(stopwords.words('english'))

    filtered = []
    for w in words:
        if w not in stop_words:
            filtered.append(w)

    return filtered

# features: string list of features
# text: string list of the text
# author_prob: priori probability of text being the current author's
def bayes_theorem_with_no_divisor(features, text, author_prob):
    pdb.set_trace()
    text_length = float(len(text))
    cond_prob = 1
    # find the conditional probability with each feature using bayes' theorem that applies Laplace smoothing
    # multiply each feature's probability with each other
    for feature in features:
        feature_count = float(text.count(feature))
        # Bayes' theorem applied with Laplace smoothing
        cond_prob = cond_prob * (((feature_count+1.0)/(text_length + possible_words)) * author_prob)
    
    return cond_prob

def file_IO(argv):
    params = {
        "training_file": 1,
        "testing_file": 2
    }
    training_file = None
    testing_file = None
    try:
        training_file = open(argv[params["training_file"]], "r")
    except IOError:
        print("Couldn't find file: " + str(argv[params["training_file"]]))
        exit(1)
    try:
        testing_file = open(argv[params["testing_file"]], "r")
    except IOError:
        print("Couldn't find file: " + str(argv[params["testing_file"]]))
        exit(1)

    return training_file, testing_file


def file_data_to_string(file):
    result = ""
    for l in file.readlines():
        result += l
    return result



if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
