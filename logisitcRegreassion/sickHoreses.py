from sklearn.linear_model import LogisticRegression

def colicSklearn():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = []; testLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabel.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabel) * 100
    print("正确率:%f%%" % test_accurcy)

if __name__ == '__main__':
    colicSklearn()