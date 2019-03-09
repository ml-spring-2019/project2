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
import glob
import pdb
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams

# priori probability of each author
hamilton_prob = 20.0/34.0
madison_prob = 14.0/34.0

def main(argv, argc):
    if (argc != 4):
        print("python main.py <Hamilton-dir> <Madison-dir> <Disputed-dir>")
        return 1

#   training_set: [[hamilton set] [jay set] [madison set]]
#   testing_set: [disputed set]
    training_set, testing_set, disputed_possible_words, author_directories = file_IO(argv)


#   testing_set: list of testing_set
#   testing_set = get_testing_set(argv[4])
#    testing_set = preprocessing(argv[4])
    bayes_scores = {}

    for i in range(len(testing_set)):
        print("Analyzing: " + author_directories[2][i] + ":")
#        print("Running Bayes Theorem with Hamilton Probability...")
        bayes_scores["Hamilton"] = bayes_theorem_with_no_divisor(testing_set[i], training_set[0], hamilton_prob, disputed_possible_words[i])

#        print("Running Bayes Theorem with Madison Probability...")
        bayes_scores["Madison"] = bayes_theorem_with_no_divisor(testing_set[i], training_set[1], madison_prob, disputed_possible_words[i])

        print_results(bayes_scores)

def print_results(bayes_scores):
    print("Bayes scores:")
    for k in bayes_scores.keys():
        print(k + ":     \t" + str(bayes_scores[k]))

    print("The author of the unknown classification is probably: " + get_author_with_highest_prob(bayes_scores) + ".\n\n")


def get_author_with_highest_prob(bayes_scores):
    highest_prob = max([bayes_scores["Hamilton"], bayes_scores["Madison"]])
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

#    pdb.set_trace()
    filtered = []
    for w in words:
        if w not in stop_words:
            filtered.append(w)

    return filtered

# testing_set: string list of testing_set
# text: string list of the text
# author_prob: priori probability of text being the current author's
def bayes_theorem_with_no_divisor(testing_file, text, author_prob, possible_words):
    text_length = float(len(text))
    cond_prob = 0.0
    smoothing_value = 1.0
    # find the conditional probability with each feature using bayes' theorem that applies Laplace smoothing
    # multiply each feature's probability with each other
    for feature in testing_file:
        feature_count = float(text.count(feature))
        # Bayes' theorem applied with Laplace smoothing

        numerator = float(feature_count + smoothing_value)
        denominator = float(text_length + possible_words)

        # using logs so that it doesn't underflow
        cond_prob += math.log((numerator / denominator))

    cond_prob += math.log(author_prob)
    
    return cond_prob

def file_IO(argv):
    print("Performing file I/O...\n\n")
    
    author_directories = []
    for i in range(1, len(argv)):
        author_directories.append(glob.glob(argv[i]+"/*.txt"))
    
    training_set = [[],[]]
    testing_set = []
    i = 0

    for directory in author_directories:
        for file in directory:
            text_file = open(file, "r")
            if i < len(author_directories)-1:
                training_set[i] += (preprocessing(text_file))
            else:
                testing_set.append(preprocessing(text_file))
        i += 1

    distinct_words = []

    for collection in training_set:
        distinct_words += list(set(collection))

    distinct_words = list(set(distinct_words))

    disputed_possible_words = {}

    for i in range(len(author_directories[2])):
        disputed_possible_words[i] = len(list(set(testing_set[i]+distinct_words)))

    return training_set, testing_set, disputed_possible_words, author_directories


def file_data_to_string(file):
    result = ""
    for l in file.readlines():
        result += l
    return result



if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
