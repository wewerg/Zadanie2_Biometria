# from itertools import permutations
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn import datasets
import math

MAX_DAT = 846

"""http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/"""


def load_file(filename):
    """Nacitanie suboru"""
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    arrays = [x.split() for x in content]
    data = [x[:18] for x in arrays]
    type = [x[18] for x in arrays]
    return data, type


def mean(numbers):
    """Vypocet strednej hodnoty"""
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    """Smerodajna odchylka"""
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    print(dataset)
    dataset = [float(i) for i in dataset]
    print(dataset)
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    #del summaries[-1]
    return summaries


def bayes(X_train, X_test, y_train, y_test):
    print()


if __name__ == "__main__":
    print("Zadanie2")
    filename = "vehicle.dat"
    data, type = load_file(filename)
    print("Loaded data file {} with {} rows".format(filename, MAX_DAT))
    # rozdelenie na trenovacie a testovacie data
    X_train, X_test, y_train, y_test = train_test_split(data, type, test_size=0.4, random_state=0)
    print(summarize(X_train[1]))