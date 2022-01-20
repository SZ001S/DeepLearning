from tokenize import group
from turtle import tilt
import numpy as np
# from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
# from os import listdir
import os 

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def createDataSet():
    group = np.array([[1.0,1.1],
                      [1.0,1.0],
                      [0,0], 
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify_0(inX, dataSet, labels, k):
    # group中的四个列表的第一个列表的大小
    dataSetSize = dataSet.shape[0] # 一列有多少元素 即多少行
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5 # 得到四个点的对应给定点的距离

    sortedDistIndicies = distances.argsort() # 返回排好序数组的对应元素下标 numpy的方法
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), 
                              key=operator.itemgetter(1), 
                              reverse=True)
    return sortedClassCount[0][0]

# group, labels = createDataSet()
# print(classify_0([0,0], group, labels, 3))

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 分布对应shape[0]和shape[1]
    returnMat = np.zeros((numberOfLines, 3)) # 共3列
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3] # 从0~2，第3列取不到
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# datingDataMat, datingLabels = file2matrix('ch02\datingTestSet2.txt')
# fig = plt.figure() # 加画布
# ax = fig.add_subplot(1,1,1)
# ax.set_ylabel("每周消费的冰淇淋公升数")
# ax.set_xlabel("玩视频游戏所耗时间百分比")
# # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], color='green')
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
# plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # 归一化
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat, '\n\n', ranges, '\n\n', minVals)

# 测试算法
def datingClassTest():
    hoRatio = 0.10 # 测试样本的比例
    datingDataMat, datingLabels = file2matrix('ch02\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] # m行
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify_0(normMat[i, :], \
                                        normMat[numTestVecs:m, :], \
                                        datingLabels[numTestVecs:m], 3)
        print(f'the classifier came back with: {classifierResult}, the real answer is: {datingLabels[i]}')
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print(f'the total error rate is: {errorCount/float(numTestVecs)}')

# # -------
# # 测试
# # -------
# datingClassTest()

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
        'percentage of time spent playing video games?'))
    ffMiles = float(input(\
        'frequent filier miles earned per year?'))
    iceCream = float(input(\
        'liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifyResult = classify_0((inArr - \
        minVals)/ranges, normMat, datingLabels, 3)
    print(f"You will probably like this person:{resultList[classifyResult - 1]}")

# classifyPerson()

def img2Vector(filename): # 转换一条数据为一个向量Vector
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect # 返回处理结果


# ---------
# 测试算法
# ---------
def handWritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList) # 有多少数据条目数
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] #以点号分隔的名字取点号前面部分
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector(f'trainingDigits/{fileNameStr}') # 返回一个行条目
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector(f'testDigits/{fileNameStr}')
        classifierResult = classify_0(vectorUnderTest,\
                                      trainingMat, hwLabels, 3)
        print(f'the classifier came back with: {classifierResult}, the real answer is: {classNumStr}')
        if (classifierResult != classNumStr) : errorCount += 1.0
    print(f'\nthe total number of errors is: {errorCount}')
    print(f'\nthe total error rate is: {errorCount/float(mTest)}')

