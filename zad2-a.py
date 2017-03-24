# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

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

def loadCsv(filename):
    """Nacitava data"""
    lines = csv.reader(open(filename, "r"), delimiter=' ')
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    """Rozdelenie na trenovacie a testovacie data"""
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    """Rozdelenie dat podla tried"""
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    """Vypocet strednej hodnoty"""
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    """odchylka"""
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    """suma"""
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    """Zosumovanie podľa tried"""
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    """Vypocet pravdepodobnosti"""
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    """Vypocet pravdepodobnosti podla tried"""
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    """Predpoved"""
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    """ziskanie predpovedi"""
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    """vypocet presnosti"""
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def getSvmValues(trainingSet, testSet):
    clf = svm.SVC()
    clf.fit()

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

    all = 0
    good = 0
    for i in range(len(y_test)):
        all += 1
        if int(y_test[i]) is int(hodnotySVM[i]):
            good += 1

    presnost = good / all

    print("presnost SVM je ", presnost)
    return presnost

def getValuesSVM(clf: object, testSet: object) -> object:
    """Získanie pravdepodobnosti pre svm"""
    predictions = []
    for i in range(len(testSet)):
        result = clf.predict(testSet[i])
        #print(result[0])
        predictions.append(result)
    return predictions

def svm_function():
    filename = "vehicle.dat"
    data, type = load_file(filename)


    data = change_matrix_to_int(data)

    # rozdelenie na trenovacie a testovacie data
    hodnotySVM = []
    splitRatios = []
    for i in range(40, 80):
        hodnotySVM.append(getSvmValue(data, type, i))
        splitRatios.append(i)
    return splitRatios, hodnotySVM

def transformSVM(hodnotySVM):
    return [x*100 for x in hodnotySVM]



def main():
    filename = 'vehicle.dat'
    dataset = loadCsv(filename)

    resultsBayes =[]
    splitRatios =[]

    for i in range(40, 80):
        splitRatio = i/100
        splitRatios.append(splitRatio)
        print(splitRatio)
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
        # prepare model
        summaries = summarizeByClass(trainingSet)
        # test model
        predictions = getPredictions(summaries, testSet)
        accuracy = getAccuracy(testSet, predictions)
        resultsBayes.append(accuracy)
        print('Accuracy: {0}%'.format(accuracy))

    splitRatiosSVM, hodnotySVM = svm_function()
    hodnotySVM = transformSVM(hodnotySVM)
    plt.plot(splitRatios, resultsBayes, "b", splitRatios, hodnotySVM, "r")
    plt.ylabel("percentualna uspesnost")
    plt.xlabel("pomer testovacich-trenovacich dat")
    plt.show()

if __name__ == "__main__":
    main()

