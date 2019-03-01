import sys
import pdb
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def main(argv, argc):
    if (argc < 3):
        print("Usage: python main.py <training-set> <testing-set>")
        return 1
    training_file, testing_file = file_IO(argv)

    training_contents = file_data_to_string(training_file)
    testing_contents = file_data_to_string(testing_file)

    filtered_training = remove_stop_words(training_contents)
    filtered_testing = remove_stop_words(testing_contents)


    pdb.set_trace()

def remove_stop_words(contents):
    words = word_tokenize(contents)
    stop_words = set(stopwords.words('english'))

    filtered = []
    for w in words:
        if w not in stop_words:
            filtered.append(w)

    return filtered


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