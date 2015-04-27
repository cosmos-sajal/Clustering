import csv
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def populateClassMap(allLabels):
	classMap = {}
	j = 0
	for i in allLabels:
		classMap[i] = j
		j = j + 1
	return classMap

'''
Loading data from the input file
into a list named dataset. The 
function then returns the list.
'''
def loadData(filePath, fileName):
	fullFilePath = filePath + "/" + fileName
	lines = csv.reader(open(fullFilePath, "rb"))
	dataset = list(lines)
	allLabels = []
	for i in range(0, len(dataset)):
		allLabels.append(dataset[i][len(dataset[0]) - 1])
	allLabels = set(allLabels)
	k = len(allLabels)
	classMap = populateClassMap(allLabels)
	actualLabels = []
	for i in range(0, len(dataset)):
		actualLabels.append(classMap[dataset[i][len(dataset[0]) - 1]])
	i = 0
	for x in dataset:
		del x[len(dataset[i]) - 1]
		i = i + 1
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset, k, actualLabels

def EuclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def minDistanceClusters(clusters,centres):
	minv=sys.float_info.max
	for i in range(0,len(clusters)):
		for j in range(0,len(clusters)):
			if i!=j:
				try:
					dist=np.subtract(centres[i],centres[j])
					norm=np.linalg.norm(dist)
					if norm<minv:
						minv=norm
						c1=i
						c2=j
				except ValueError:
					print "Error"
					print centres[i]
					print centres[j]
	#print minv,c1,c2
	return c1,c2

def calculateMean(clusters):
	centres=[]
	for i in range(0,len(clusters)):	
		centres.append([])
		centres[i] = list(np.mean(clusters[i],axis=0))
	return centres

def Hierarichal(data, actualLabels, k):
	clusters=[]
	for line in data:
		clusters.append([line])
	print len(clusters)
	centres=calculateMean(clusters)
	while len(clusters) != k:
		c1,c2 = minDistanceClusters(clusters,centres)
		clusters[c1] = clusters[c1]+clusters[c2]		
		clusters.remove(clusters[c2])
		centres = calculateMean(clusters)
	return clusters, centres

def calPosOfTheDataPointInData(dataPoint, data):
	pos = 0
	for i in range(0, len(data)):
		if dataPoint == data[i]:
			pos = i
			break
	return pos

def calPredictedLabels(clusters, data, k):
	predictedLabels = [0 for x in range(len(data))]
	for i in range(0, k):
		for j in range(0, len(clusters[i])):
			dataPoint = clusters[i][j]
			pos = calPosOfTheDataPointInData(dataPoint, data)
			predictedLabels[pos] = i
	return predictedLabels

def calculateCorrelation(actualLabels, predictedLabels):
	array1 = []
	array2 = []
	for i in range(0, len(actualLabels)):
		for j in range(0, len(actualLabels)):
			if actualLabels[i] == actualLabels[j]:
				array1.append(1)
			else:
				array1.append(0)
	for i in range(0, len(predictedLabels)):
		for j in range(0, len(predictedLabels)):
			if predictedLabels[i] == predictedLabels[j]:
				array2.append(1)
			else:
				array2.append(0)
	correlationMatrix = np.corrcoef(array1, array2)
	return correlationMatrix[0][1]

def generateRandomRGB():
	hexDigits = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
	r1 = random.choice(hexDigits)
	r2 = random.choice(hexDigits)
	g1 = random.choice(hexDigits)
	g2 = random.choice(hexDigits)
	b1 = random.choice(hexDigits)
	b2 = random.choice(hexDigits)
	randomColour = "#" + str(r1) + str(r2) + str(g1) + str(g2) + str(b1) + str(b2)
	return randomColour

def plotResults(actualLabels, predictedLabels, data, k):
	# Plotting actual labels
	colours = []
	for i in range(0, k):
		X = []
		Y = []
		for j in range(0, len(data)):
			if actualLabels[j] == i:
				X.append(data[j][0])
				Y.append(data[j][1])
		randomColour = generateRandomRGB()
		colours.append(randomColour)
		plt.subplot(2,1,1)
		plt.plot(X, Y, randomColour, linestyle=':')
		plt.ylabel('Actual', fontsize=20)
	# Plotting predicted labels
	for i in range(0, k):
		X = []
		Y = []
		for j in range(0, len(data)):
			if predictedLabels[j] == i:
				X.append(data[j][0])
				Y.append(data[j][1])
		plt.subplot(2,1,2)
		plt.plot(X, Y, colours[i], linestyle=':')
		plt.ylabel('Predicted', fontsize=20)
	plt.show()

def calRootMeanSquaredError(centroids, data, predictedLabels):
	rootMeanSquaredError = 0.0
	for i in range(0, len(data)):
		rootMeanSquaredError = rootMeanSquaredError + EuclideanDistance(centroids[predictedLabels[i]], data[i], len(data[0]))
	return rootMeanSquaredError

def calDunnIndex(data, centroids, predictedLabels, k):
	maxDistancesInCluster = []
	for i in range(0, k):
		tempData = []
		for j in range(0, len(data)):
			if predictedLabels[j] == i:
				tempData.append(data[j])
		maxDistance = -sys.maxint
		for l in range(0, len(tempData)):
			for m in range(0, len(tempData)):
				temp = EuclideanDistance(tempData[l], tempData[m], len(tempData[0]))
				if temp > maxDistance:
					maxDistance = temp
		maxDistancesInCluster.append(maxDistance)
	denominator = max(maxDistancesInCluster)
	minDistance = sys.maxint
	for i in range(0, len(centroids)):
		for j in range(0, len(centroids)):
			if i != j:
				temp = EuclideanDistance(centroids[i], centroids[j], len(centroids[0]))
				if temp < minDistance:
					minDistance = temp
	numerator = minDistance
	dunnIndex = float(numerator / denominator)
	return dunnIndex

def calPurity(actualLabels, predictedLabels, k):
	purity = 0.0
	for i in range(0, k):
		noOfLabelsOfEachClassInTheCluster = [0 for x in range(0, k)]
		if i == 0:
			startPoint = 0
			for j in range(1, len(actualLabels)):
				if actualLabels[j] != actualLabels[j - 1]:
					endPoint = j - 1
					break
			for j in range(startPoint, endPoint + 1):
				noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] = noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] + 1
			purity = purity + max(noOfLabelsOfEachClassInTheCluster)
		elif i == k - 1:
			startPoint = endPoint + 1
			endPoint = len(actualLabels) - 1
			for j in range(startPoint, endPoint + 1):
				noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] = noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] + 1
			purity = purity + max(noOfLabelsOfEachClassInTheCluster)
		else:
			startPoint = endPoint + 1
			for j in range(startPoint + 1, len(actualLabels)):
				if actualLabels[j] != actualLabels[j - 1]:
					endPoint = j - 1
					break
			for j in range(startPoint, endPoint + 1):
				noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] = noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] + 1
			purity = purity + max(noOfLabelsOfEachClassInTheCluster)
	purity = float(purity / len(actualLabels))
	return purity

def calMapping(actualLabels, predictedLabels, k):
	labelMap = {}
	for i in range(0, k):
		noOfLabelsOfEachClassInTheCluster = [0 for x in range(0, k)]
		if i == 0:
			startPoint = 0
			for j in range(1, len(actualLabels)):
				if actualLabels[j] != actualLabels[j - 1]:
					endPoint = j - 1
					break
			for j in range(startPoint, endPoint + 1):
				noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] = noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] + 1
		elif i == k - 1:
			startPoint = endPoint + 1
			endPoint = len(actualLabels) - 1
			for j in range(startPoint, endPoint + 1):
				noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] = noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] + 1
		else:
			startPoint = endPoint + 1
			for j in range(startPoint + 1, len(actualLabels)):
				if actualLabels[j] != actualLabels[j - 1]:
					endPoint = j - 1
					break
			for j in range(startPoint, endPoint + 1):
				noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] = noOfLabelsOfEachClassInTheCluster[predictedLabels[j]] + 1
		maxVal = -sys.maxint
		for j in range(0, len(noOfLabelsOfEachClassInTheCluster)):
			temp = noOfLabelsOfEachClassInTheCluster[j]
			if temp > maxVal:
				maxVal = temp
				pos = j
		labelMap[pos] = actualLabels[startPoint]
	return labelMap

def calConfusionMatrix(actualLabels, predictedLabels, k):
	labelMap = calMapping(actualLabels, predictedLabels, k)
	confusionMatrix = [[0 for x in range(k)] for x in range(k)]
	for i in range(0, len(actualLabels)):
		try:
			confusionMatrix[int(actualLabels[i])][int(labelMap[predictedLabels[i]])] = confusionMatrix[int(actualLabels[i])][int(labelMap[predictedLabels[i]])] + 1
		except KeyError, e:
			continue
	return confusionMatrix

def printConfusionMatrix(confusionMatrix):
	print "Confusion Matrix:"
	for i in range(0, len(confusionMatrix)):
		print confusionMatrix[i]

def printResults(correlation, actualLabels, predictedLabels, centroids, data, k):
	print "Actual Labels:", actualLabels
	print "Predicted Labels:", predictedLabels
	print "External Measure(correlation):", correlation
	purity = calPurity(actualLabels, predictedLabels, k)
	print "External Measure(Purity):", purity
	rootMeanSquaredError = calRootMeanSquaredError(centroids, data, predictedLabels)
	print "Internal Measure(Root Mean Squared Error):", rootMeanSquaredError
	dunnIndex = calDunnIndex(data, centroids, predictedLabels, k)
	print "Internal Measure(Dunn Index):", dunnIndex
	confusionMatrix = calConfusionMatrix(actualLabels, predictedLabels, k)
	printConfusionMatrix(confusionMatrix)
	plotResults(actualLabels, predictedLabels, data, k)

def main(arg):
	filePath = str(arg[0])
	fileName = str(arg[1])
	data, k, actualLabels = loadData(filePath, fileName)
	clusters, centroids = Hierarichal(data, actualLabels, k)
	predictedLabels = calPredictedLabels(clusters, data, k)
	correlation = calculateCorrelation(actualLabels, predictedLabels)
	printResults(correlation, actualLabels, predictedLabels, centroids, data, k)

main(sys.argv[1:])