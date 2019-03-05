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
import math
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
        print("Usage: python main.py <known-author-dir> <unknown-author-dir>")
        return 1
    training_set, testing_set = file_IO(argv)

    bayes_scores = {}

    print("Running Bayes Theorem with Hamilton Probability...")
    bayes_scores["hamilton"] = bayes_theorem_with_no_divisor(testing_set, training_set, hamilton_prob)

    print("Running Bayes Theorem with Jay Probability...")
    bayes_scores["jay"] = bayes_theorem_with_no_divisor(testing_set, training_set, jay_prob)

    print("Running Bayes Theorem with Madison Probability...")
    bayes_scores["madison"] = bayes_theorem_with_no_divisor(testing_set, training_set, madison_prob)

    print_results(bayes_scores)

    pdb.set_trace()

def print_results(bayes_scores):
    print("Bayes scores:")
    for k in bayes_scores.keys():
        print(k + ":     \t" + str(bayes_scores[k]))

    print("The author of the unknown classification is probably " + get_author_with_highest_prob(bayes_scores) + ".")


def get_author_with_highest_prob(bayes_scores):
    highest_prob = max([bayes_scores["hamilton"], bayes_scores["jay"], bayes_scores["madison"]])
    for k in bayes_scores.keys():
        if bayes_scores[k] == highest_prob:
            return k
    return None


def preprocessing(file):
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
    text_length = float(len(text))
    cond_prob = 0.0
    smoothing_value = 1.0
    # find the conditional probability with each feature using bayes' theorem that applies Laplace smoothing
    # multiply each feature's probability with each other
    for feature in features:
        feature_count = float(text.count(feature))
        # Bayes' theorem applied with Laplace smoothing

        numerator = float(feature_count + smoothing_value)
        denominator = float(text_length + possible_words)

        cond_prob += math.log((numerator / denominator) * author_prob)

    return cond_prob

def file_IO(argv):
    print("Performing file I/O...")

    params = {
        "training_dir": 1,
        "testing_dir": 2
    }
    training_file = None
    testing_file = None

    training_set = []
    testing_set = []
    files_not_found = 0
    files_found_training = 0
    files_found_testing = 0

    for i in range(0, 100):
        try:
            training_filename = argv[params["training_dir"]] + str(i) + ".txt"
            training_file = open(training_filename, "r")
            training_set += preprocessing(training_file)
            files_found_training += 1
        except IOError:
            files_not_found += 1
        try:
            testing_filename = argv[params["testing_dir"]] + str(i) + ".txt"
            testing_file = open(testing_filename, "r")
            testing_set += preprocessing(testing_file)
            files_found_testing += 1
        except IOError:
            files_not_found += 1

    if files_found_training == 0:
        print("No files found for building the training set. Please check if the directory is correct and files range from 0.txt to 100.txt.")
        exit(1)
    elif files_found_testing == 0:
        print("No files found for building the testing set. Please check if the directory is correct and files range from 0.txt to 100.txt.")
        exit(1)

    return training_set, testing_set


def file_data_to_string(file):
    result = ""
    for l in file.readlines():
        result += l
    return result



if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
