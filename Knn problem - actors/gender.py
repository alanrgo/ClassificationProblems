from pylab import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import scipy
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import math

class LabeledImg:
    def __init__(self, name, img, gender = 'none'):
        self.name = name
        self.img = img
        self.genderLabel = gender
        
class LabeledNumber:
    def __init__(self, number, label):
        self.number = number
        self.name = label
        
def kNN(k, lblimg, trainingSet):
    distanceList = []
    
    actRanking = [ LabeledNumber(0, 'male'), LabeledNumber(0, 'female')]
    
    for index in range(0, len(trainingSet)):
        sum = 0
        for i in range (0, 32):
            for j in range(0, 32):
                distance = lblimg.img[i, j]-trainingSet[index].img[i, j]
                distance = distance * distance
                sum += distance
        distanceList.append(LabeledNumber(math.sqrt(sum), trainingSet[index].name))
    distanceList = sorted(distanceList, key=lambda distance: distance.number)
    
    #print "nextlist"
    #for i in range(0, len(distanceList)):
    #    print distanceList[i].dist, distanceList[i].name
    
    
    for i in range(0,k):
        if distanceList[i].name == 'butler':
            actRanking[0].number += 1
        elif distanceList[i].name == 'radcliffe':
            actRanking[0].number += 1
        elif distanceList[i].name == 'vartan':
            actRanking[0].number += 1
        elif distanceList[i].name == 'bracco':
            actRanking[1].number += 1
        elif distanceList[i].name == 'gilpin':
            actRanking[1].number += 1
        elif distanceList[i].name == 'harmon':
            actRanking[1].number += 1
    
    actRanking = sorted(actRanking, key=lambda act: act.number)
    for a in actRanking:
        print a.name, a.number
    
    print 'The picture is: ' + lblimg.name + ' Gender: ' + actRanking[1].name
    
    if lblimg.genderLabel == actRanking[1].name:
        return True
    return False

'''
act_female = list(set([a.split("\t")[0].split()[-1].lower() for a in open('./Python Scripts/CSC321/subset_actresses.txt').readlines()]))
act_male = list(set([a.split("\t")[0].split()[-1].lower() for a in open('./Python Scripts/CSC321/subset_actors.txt').readlines()]))

for a in act_female:
    print a
    
for a in act_male:
    print a
    '''

act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
    

for i in range (0, len(act)):
    act[i] = act[i].split()[1].lower()
    print act[i]

filelist = os.listdir('training_set')
print filelist

trainingSet = []
validatedSet = []

for i in range(0, len(filelist)):
    for j in range (0, len(act)):
        if act[j] in filelist[i]:
            try:
                if j < 3:
                    trainingSet.append(LabeledImg(act[j], scipy.misc.imread('training_set/'+filelist[i]), 'male')) #append a Labeled obj
                else:
                    trainingSet.append(LabeledImg(act[j], scipy.misc.imread('training_set/'+filelist[i]), 'female')) #append a Labeled obj
            except:
                print 'Erro1'
            break

filelist2 = os.listdir('validation_set')
print filelist2

correct = 0
total = 0

butler = 0
radc = 0
bracco = 0
vartan = 0
harmon = 0
gilpin = 0

kResults = []

for k in range(1, 15):
    tempNbr = LabeledNumber(0, str(k) )
    kResults.append(tempNbr)

for i in range(0, len(filelist2)):
    #load the images
    total += 1
    for j in range (0, len(act)):
        if act[j] in filelist2[i]:
            
            temp_img = scipy.misc.imread('validation_set/'+filelist2[i])
            if j < 3:
                temp_labeledObj = LabeledImg(act[j], temp_img, 'male')
            else:
                temp_labeledObj = LabeledImg(act[j], temp_img, 'female')
            validatedSet.append(temp_labeledObj) #append a Labeled obj
            
            for k in range(1, 15):
                #calculate all the distances
                if kNN(k, temp_labeledObj, trainingSet):
                    kResults[k-1].number += 1
                    '''if temp_labeledObj.name == 'butler':
                        butler += 1
                    elif temp_labeledObj.name == 'radcliffe':
                        radc += 1
                    elif temp_labeledObj.name == 'bracco':
                        bracco += 1
                    elif temp_labeledObj.name == 'gilpin':
                        gilpin += 1
                    elif temp_labeledObj.name == 'harmon':
                        harmon += 1
                    elif temp_labeledObj.name == 'vartan':
                        vartan += 1'''
            
            break
            
    #calculate all the distances
    #sort the distance values, preserving a key for each image
'''print correct, total
print 'butler = ' + str(butler)
print 'radcliffe = ' + str(radc)
print 'bracco = ' + str(bracco)
print 'gilpin = ' + str(gilpin)
print 'harmon = ' + str(harmon)
print 'vartan = ' + str(vartan)'''

x = []
y = []
for k in range(0, 14):
    temp = kResults[k].number/float(60)
    x.append(k)
    y.append(temp)
    print 'k = ' + str(kResults[k].number) + ' Perc: ' + str(temp)
plt.plot(x, y, 'bo')
plt.xlabel('K values')
plt.ylabel('Accuracy of kNN algorithm due to the value of K')  
plt.savefig('Accuracy_K_plot1.png')
plt.close()

print float(correct)/float(total)