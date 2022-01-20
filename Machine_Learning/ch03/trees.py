from math import log 
import math
import operator 
# import numpy as np

def createDataSet():
    dataSet = [[1, 1, 'yes'], 
               [1, 1, 'yes'], 
               [1, 0, 'no'], 
               [0, 1, 'no'], 
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVect in dataSet:
        currentLabel = featVect[-1] # 最后一个条目
        # print(featVect[-1])
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        # print(labelCounts)    
    # 计算部分
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2) # shannonEnt为负值，累加就是累减
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    >>> trees.splitDataSet(myDat, 0, 1)
    []
    [1, 'yes']
    [[1, 'yes']]

    []
    [1, 'yes']
    [[1, 'yes'], [1, 'yes']]

    []
    [0, 'no']
    [[1, 'yes'], [1, 'yes'], [0, 'no']]

    [[1, 'yes'], [1, 'yes'], [0, 'no']]
    >>>
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatureVec = featVec[:axis]
            # print(reducedFeatureVec)
            reducedFeatureVec.extend(featVec[axis+1:])
            # print(reducedFeatureVec)
            retDataSet.append(reducedFeatureVec)
            # print(retDataSet)
            # print()
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    >>> trees.chooseBestFeatureToSplit(myDat)
    [1, 1, 'yes']   
    2
    [1, 1, 1, 0, 0]
    {0, 1}
    [[1, 'no'], [1, 'no']]
    [[1, 'yes'], [1, 'yes'], [0, 'no']]

    [1, 1, 0, 1, 1]
    {0, 1}
    [[1, 'no']]
    [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]

    0
    """
    numFeatures = len(dataSet[0]) - 1
    # print(dataSet[0])
    # print(numFeatures)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures): # 从0到1 表示标签['yes', 'no']不算其中
        featList = [example[i] for example in dataSet] # 两个不同列数据
        # print(featList)
        uniqueVals = set(featList)
        # print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # print(subDataSet)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) # 负值
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        # print()
    return bestFeature

# 构建决策树
def majorityCnt(classList):
    classCount = {}
    # print(classList)
    for vote in classList:
        if vote not in classList.keys(): classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), 
                                  key=operator.itemgetter(1), 
                                  reverse=True)
        # print(sortedClassCount)
        # print("majorityCnt部分")
        return sortedClassCount[0][0] # 键值对

def createTree(dataSet, labels):
    print("===================")
    classList = [example[-1] for example in dataSet]
    print(classList)
    print(classList[0])
    print(classList.count(classList[0]))
    if classList.count(classList[0]) == len(classList): 
        print('***')
        return classList[0]     # 类别相同，就是纯度最高
    print(dataSet[0], len(dataSet[0]))
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print(bestFeat)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    print(featValues)
    uniqueVals = set(featValues)
    print(uniqueVals)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
                                                  splitDataSet(dataSet, 
                                                               bestFeat, 
                                                               value), 
                                                  subLabels)                                                  
    return myTree

