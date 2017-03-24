# from itertools import permutations
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn import datasets
import math
import matplotlib.pyplot as plt
import pprint
from sklearn import svm

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
    print("cislo {}".format(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    # del summaries[-1]
    return summaries


def bayes(X_train, X_test, y_train, y_test):
    print()


def calculateProbability(x, mean, stdev):
    """Vypocet pravdepodobnosti"""
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    """pravdepodobnosti tried"""
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    """Získanie pravdepodobnosti"""
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getValuesSVM(clf, testSet):
    """Získanie pravdepodobnosti pre svm"""
    predictions = []
    for i in range(len(testSet)):
        result = clf.predict(testSet[i])
        #print(result[0])
        predictions.append(result)
    return predictions


def change_values_to_int(input):
    pole = [int(i) for i in input]
    return pole


def change_matrix_to_int(matrix):
    matrix = [change_values_to_int(i) for i in matrix]
    return matrix


def getSvmValue(data, type, splitRatio):
    number_of_test = 1 - splitRatio
    X_train, X_test, y_train, y_test = train_test_split(data, type, test_size=number_of_test, random_state=1)
    y_train = change_values_to_int(y_train)
    X = np.array(X_train)
    y = y_train

    """natrenovanie SVM"""
    clf = svm.SVC(kernel="linear", C=2.0)
    clf.fit(X, y)
    hodnotySVM = getValuesSVM(clf, X_test)
    # hodnotySVM, y_test
    good_values = []
    all = 0
    good = 0
    for i in range(len(y_test)):
        all += 1
        if int(y_test[i]) is int(hodnotySVM[i]):
            good += 1

    presnost = good / all

    print("presnost SVM je ", presnost)
    return presnost


if __name__ == "__main__":
    print("Zadanie2")
    filename = "vehicle.dat"
    data, type = load_file(filename)
    print("Loaded data file {} with {} rows".format(filename, MAX_DAT))

    data = change_matrix_to_int(data)

    # rozdelenie na trenovacie a testovacie data
    hodnotySVM = []
    splitRatios = []
    for i in range(40, 80):
        hodnotySVM.append(getSvmValue(data, type, i))
        splitRatios.append(i)

    plt.plot(splitRatios, hodnotySVM, "r")
    plt.show()
