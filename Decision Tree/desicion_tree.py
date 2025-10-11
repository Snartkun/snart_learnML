from math import log
from tkinter.scrolledtext import example


def creatDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']  # 分类属性
    return dataSet, labels

def calcShannonEnt(dataset):
    nums = len(dataset)
    LabelCount = {}
    for feaVec in dataset:
        curLabel = feaVec[-1]
        if curLabel not in LabelCount.keys():
            LabelCount[curLabel] = 0
        LabelCount[curLabel] += 1
    shannonEnt = 0.0
    for key in LabelCount:
        prob = float(LabelCount[key]) / nums
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSpilit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeatures = -1
    for i in range(numFeatures):
        features = [example[i] for example in dataSet]
        uniqueList = set(features)
        newEntropy = 0.0
        for value in uniqueList:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatures = i
    return bestFeatures

if __name__ == '__main__':
    dataSet, features = creatDataSet()
    print("最优索引值：" + str(chooseBestFeatureToSpilit(dataSet)))