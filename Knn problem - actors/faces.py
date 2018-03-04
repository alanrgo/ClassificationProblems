import sys
from pylab import *
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
import shutil
from scipy.ndimage import filters
import urllib
from PIL import Image


class LabeledImg:
    def __init__(self, name, img, filename, gender = 'none'):
        '''Labeled Image is a class which permits you to see the name of the actor,
        the image in numpy, and the name of the file'''
        self.name = name
        self.img = img
        self.genderLabel = gender
        self.filename = filename
        
class LabeledNumber:
    '''Labeled Number is used to create a label to the specific value'''
    def __init__(self, number, label):
        self.number = number
        self.name = label

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    
    
    if it.isAlive():
        return False
    else:
        #print type(it)
        #For a reason that I do not know yet, this function keeps trying to see
        #if the download is working
        return it.result
        
def OrganizeTrainingSet():
    #files to move
    fileToMove = ['bracco0.jpg', 'bracco1.jpg', 'bracco10.jpg', 'bracco100.jpg', 'bracco101.jpg', 'bracco11.jpg', 'bracco12.jpg', 
'bracco13.jpg', 'bracco14.jpg', 'bracco15.jpg', 'bracco16.jpg', 'bracco17.jpg', 'bracco19.jpg', 'bracco2.jpg', 
'bracco20.jpg', 'bracco21.jpg', 'bracco22.jpg', 'bracco23.jpg', 'bracco24.jpg', 'bracco25.jpg', 'bracco26.jpg', 
'bracco28.jpg', 'bracco29.jpeg', 'bracco3.jpg', 'bracco31.jpg', 'bracco37.jpg', 'bracco38.jpg', 'bracco39.jpg', 
'bracco4.jpg', 'bracco40.jpg', 'bracco41.jpg', 'bracco42.jpg', 'bracco43.jpg', 'bracco44.jpg', 'bracco45.jpg', 
'bracco46.jpg', 'bracco48.jpg', 'bracco49.jpg', 'bracco5.jpg', 'bracco50.jpg', 'bracco51.jpg', 'bracco52.jpg', 
'bracco53.jpg', 'bracco54.JPG', 'bracco55.jpg', 'bracco56.jpg', 'bracco57.jpg', 'bracco58.jpg', 'bracco59.jpg', 
'bracco6.jpg', 'bracco60.jpg', 'bracco61.jpg', 'bracco62.jpg', 'bracco64.jpg', 'bracco65.jpg', 'bracco66.jpg', 
'bracco68.jpg', 'bracco69.jpg', 'bracco7.jpg', 'bracco70.jpg', 'bracco71.jpg', 'bracco72.jpg', 'bracco73.jpg', 
'bracco74.jpg', 'bracco75.jpg', 'bracco76.jpg', 'bracco78.jpg', 'bracco79.jpg', 'bracco8.jpg', 'bracco80.jpg', 
'bracco81.jpg', 'bracco82.jpg', 'bracco83.jpg', 'bracco84.jpg', 'bracco85.jpg', 'bracco86.jpg', 'bracco87.jpg', 
'bracco88.jpg', 'bracco89.jpg', 'bracco9.jpg', 'bracco90.jpg', 'bracco91.jpg', 'bracco92.jpg', 'bracco93.jpg', 
'bracco94.jpg', 'bracco95.jpg', 'bracco96.jpg', 'bracco97.png', 'bracco98.jpg', 'bracco99.jpg', 'butler0.jpg', 
'butler1.jpeg', 'butler10.jpg', 'butler11.jpg', 'butler12.jpg', 'butler13.jpg', 'butler14.jpg', 'butler15.jpg', 
'butler16.jpg', 'butler17.jpg', 'butler18.jpg', 'butler19.jpg', 'butler2.jpg', 'butler20.jpg', 'butler21.jpg', 
'butler22.jpg', 'butler23.jpg', 'butler24.jpg', 'butler25.jpg', 'butler26.jpg', 'butler27.jpg', 'butler28.jpg', 
'butler29.jpg', 'butler3.jpg', 'butler30.jpg', 'butler31.jpg', 'butler32.jpg', 'butler33.jpg', 'butler34.jpg', 
'butler35.jpg', 'butler36.jpg', 'butler37.jpg', 'butler38.jpg', 'butler39.jpg', 'butler4.jpg', 'butler40.jpg', 
'butler41.jpg', 'butler42.jpg', 'butler43.jpg', 'butler44.jpg', 'butler45.jpg', 'butler46.jpg', 'butler47.jpg', 
'butler48.jpg', 'butler49.jpg', 'butler5.jpg', 'butler50.jpg', 'butler51.jpg', 'butler52.jpg', 'butler53.jpg', 
'butler54.jpg', 'butler55.jpg', 'butler56.jpg', 'butler57.jpg', 'butler58.jpg', 'butler59.jpg', 'butler6.jpg', 
'butler60.jpg', 'butler61.jpg', 'butler62.jpg', 'butler63.jpg', 'butler64.jpg', 'butler65.jpg', 'butler66.jpg', 
'butler67.jpg', 'butler68.jpg', 'butler69.jpg', 'butler7.jpg', 'butler70.jpg', 'butler71.jpg', 'butler72.jpg', 
'butler74.jpg', 'butler75.jpg', 'butler76.jpg', 'butler77.jpg', 'butler78.jpg', 'butler79.jpg', 'butler8.jpg', 
'butler80.jpg', 'butler81.jpg', 'butler82.jpg', 'butler83.jpg', 'butler84.jpg', 'butler85.jpg', 'butler86.jpg', 
'butler87.jpg', 'butler88.jpg', 'butler89.jpg', 'butler9.jpg', 'butler90.jpg', 'gilpin0.jpg', 'gilpin1.jpg', 
'gilpin10.jpg', 'gilpin11.jpg', 'gilpin12.jpg', 'gilpin13.jpg', 'gilpin14.jpg', 'gilpin15.jpg', 'gilpin16.jpg', 
'gilpin17.jpg', 'gilpin18.jpg', 'gilpin19.jpg', 'gilpin2.jpg', 'gilpin20.jpg', 'gilpin21.jpg', 'gilpin22.jpg', 
'gilpin23.jpg', 'gilpin24.jpg', 'gilpin25.jpg', 'gilpin26.JPG', 'gilpin27.jpg', 'gilpin28.jpg', 'gilpin29.jpg', 
'gilpin3.jpg', 'gilpin30.jpg', 'gilpin31.jpg', 'gilpin32.jpg', 'gilpin33.jpg', 'gilpin34.jpg', 'gilpin35.jpg', 
'gilpin36.jpg', 'gilpin37.jpg', 'gilpin38.jpg', 'gilpin39.jpg', 'gilpin4.jpg', 'gilpin40.jpg', 'gilpin41.jpg', 
'gilpin43.jpg', 'gilpin44.jpg', 'gilpin45.jpg', 'gilpin46.jpg', 'gilpin47.jpg', 'gilpin48.jpg', 'gilpin49.jpg', 
'gilpin5.jpg', 'gilpin50.jpg', 'gilpin51.jpg', 'gilpin52.jpg', 'gilpin53.jpg', 'gilpin54.jpg', 'gilpin55.jpg', 
'gilpin56.jpg', 'gilpin57.jpg', 'gilpin58.jpg', 'gilpin59.jpg', 'gilpin6.jpg', 'gilpin60.jpg', 'gilpin61.jpg', 
'gilpin62.jpg', 'gilpin63.jpg', 'gilpin64.jpg', 'gilpin65.jpg', 'gilpin66.jpg', 'gilpin67.jpg', 'gilpin68.jpg', 
'gilpin69.jpg', 'gilpin7.jpg', 'gilpin70.jpg', 'gilpin71.jpg', 'gilpin72.jpg', 'gilpin73.jpg', 'gilpin74.jpg', 
'gilpin75.jpg', 'gilpin76.jpg', 'gilpin77.jpg', 'gilpin78.jpg', 'gilpin79.jpg', 'gilpin8.jpg', 'gilpin80.jpg', 
'gilpin81.jpg', 'gilpin82.jpg', 'gilpin83.jpg', 'gilpin84.jpg', 'gilpin85.jpg', 'gilpin86.jpg', 'gilpin87.jpg', 
'gilpin88.jpg', 'gilpin89.jpg', 'gilpin9.jpg', 'gilpin90.jpg', 'harmon0.jpg', 'harmon1.jpg', 'harmon10.jpg', 
'harmon11.jpg', 'harmon12.jpg', 'harmon13.jpg', 'harmon14.jpg', 'harmon15.jpg', 'harmon16.jpg', 'harmon17.jpg', 
'harmon18.jpg', 'harmon19.jpg', 'harmon2.jpg', 'harmon20.jpg', 'harmon21.jpg', 'harmon22.jpg', 'harmon23.jpg', 
'harmon24.jpg', 'harmon25.jpg', 'harmon26.jpg', 'harmon27.jpg', 'harmon28.jpg', 'harmon29.jpg', 'harmon3.jpg', 
'harmon30.jpg', 'harmon31.png', 'harmon32.jpg', 'harmon33.jpg', 'harmon34.jpg', 'harmon35.jpg', 'harmon36.jpg', 
'harmon37.jpg', 'harmon38.jpg', 'harmon39.jpg', 'harmon4.jpg', 'harmon40.jpg', 'harmon41.jpg', 'harmon42.jpg', 
'harmon43.jpg', 'harmon44.jpg', 'harmon45.jpg', 'harmon46.jpg', 'harmon47.jpg', 'harmon48.jpg', 'harmon49.jpg', 
'harmon5.jpg', 'harmon50.jpg', 'harmon51.jpg', 'harmon52.jpg', 'harmon53.jpg', 'harmon54.jpg', 'harmon55.jpg', 
'harmon56.png', 'harmon57.jpg', 'harmon58.jpg', 'harmon59.jpg', 'harmon6.jpg', 'harmon60.jpg', 'harmon61.jpg', 
'harmon62.jpg', 'harmon63.jpg', 'harmon64.jpg', 'harmon65.jpg', 'harmon67.jpg', 'harmon68.jpg', 'harmon7.jpg', 
'harmon71.jpg', 'harmon72.jpg', 'harmon73.jpg', 'harmon74.jpg', 'harmon75.jpg', 'harmon76.jpg', 'harmon77.jpg', 
'harmon79.jpg', 'harmon8.jpg', 'harmon80.jpg', 'harmon81.jpg', 'harmon82.jpg', 'harmon83.jpg', 'harmon84.jpg', 
'harmon85.jpg', 'harmon86.jpg', 'harmon88.jpg', 'harmon89.jpg', 'harmon9.jpg', 'harmon91.jpg', 'harmon93.jpg', 
'harmon94.jpg', 'harmon95.jpg', 'harmon96.jpg', 'radcliffe0.jpg', 'radcliffe1.jpg', 'radcliffe10.jpg', 'radcliffe101.jpg', 
'radcliffe102.jpg', 'radcliffe103.jpg', 'radcliffe104.jpg', 'radcliffe105.jpg', 'radcliffe106.jpg', 'radcliffe11.jpg', 'radcliffe12.jpg', 
'radcliffe13.jpg', 'radcliffe14.jpg', 'radcliffe15.jpg', 'radcliffe16.jpg', 'radcliffe17.jpg', 'radcliffe19.jpg', 'radcliffe2.jpg', 
'radcliffe20.jpg', 'radcliffe21.jpg', 'radcliffe26.jpg', 'radcliffe27.jpg', 'radcliffe28.jpg', 'radcliffe29.jpg', 'radcliffe30.jpg', 
'radcliffe31.jpg', 'radcliffe36.jpg', 'radcliffe37.jpg', 'radcliffe38.jpg', 'radcliffe39.jpg', 'radcliffe4.jpg', 'radcliffe40.jpg', 
'radcliffe41.jpg', 'radcliffe42.jpg', 'radcliffe43.jpg', 'radcliffe44.jpg', 'radcliffe46.jpg', 'radcliffe47.jpg', 'radcliffe48.jpg', 
'radcliffe5.jpg', 'radcliffe50.jpg', 'radcliffe51.jpg', 'radcliffe52.jpg', 'radcliffe54.jpg', 'radcliffe56.jpg', 'radcliffe57.jpg', 
'radcliffe58.jpg', 'radcliffe59.jpg', 'radcliffe60.jpg', 'radcliffe61.jpg', 'radcliffe62.jpg', 'radcliffe63.jpg', 'radcliffe64.jpg', 
'radcliffe65.jpg', 'radcliffe66.jpg', 'radcliffe67.jpg', 'radcliffe68.jpg', 'radcliffe69.jpg', 'radcliffe70.jpg', 'radcliffe71.jpg', 
'radcliffe72.jpg', 'radcliffe73.jpg', 'radcliffe74.jpg', 'radcliffe75.jpg', 'radcliffe76.jpg', 'radcliffe77.jpg', 'radcliffe78.jpg', 
'radcliffe79.jpg', 'radcliffe8.jpg', 'radcliffe80.jpg', 'radcliffe81.jpg', 'radcliffe82.jpg', 'radcliffe83.jpg', 'radcliffe84.jpeg', 
'radcliffe85.jpg', 'radcliffe86.jpg', 'radcliffe87.jpg', 'radcliffe88.jpg', 'radcliffe89.jpg', 'radcliffe9.jpg', 'radcliffe90.jpg', 
'radcliffe91.jpg', 'radcliffe92.jpg', 'radcliffe93.jpg', 'radcliffe94.jpg', 'radcliffe95.jpg', 'radcliffe96.jpg', 'radcliffe97.jpg', 
'radcliffe98.jpeg', 'radcliffe99.jpg', 'vartan0.jpg', 'vartan1.JPG', 'vartan10.jpg', 'vartan11.jpg', 'vartan12.jpg', 
'vartan13.jpg', 'vartan14.jpg', 'vartan15.jpg', 'vartan16.jpg', 'vartan17.jpg', 'vartan18.jpg', 'vartan19.jpg', 
'vartan2.jpg', 'vartan20.jpg', 'vartan21.jpg', 'vartan22.jpg', 'vartan23.jpg', 'vartan24.jpg', 'vartan25.jpg', 
'vartan26.jpg', 'vartan27.jpg', 'vartan28.jpg', 'vartan29.jpg', 'vartan3.jpg', 'vartan30.jpg', 'vartan31.jpg', 
'vartan32.jpg', 'vartan33.jpg', 'vartan34.jpg', 'vartan35.jpg', 'vartan36.jpg', 'vartan37.jpg', 'vartan38.jpg', 
'vartan39.jpg', 'vartan4.jpg', 'vartan40.jpg', 'vartan41.jpg', 'vartan42.jpg', 'vartan43.jpg', 'vartan44.jpg', 
'vartan45.jpg', 'vartan46.jpg', 'vartan47.jpg', 'vartan48.jpg', 'vartan49.jpg', 'vartan5.jpg', 'vartan50.jpg', 
'vartan51.jpg', 'vartan52.jpg', 'vartan53.jpg', 'vartan54.jpg', 'vartan55.jpg', 'vartan56.jpg', 'vartan57.jpg', 
'vartan58.jpg', 'vartan59.jpg', 'vartan6.jpg', 'vartan60.jpg', 'vartan61.jpg', 'vartan62.jpg', 'vartan63.jpg', 
'vartan64.jpg', 'vartan65.jpg', 'vartan66.jpg', 'vartan67.jpg', 'vartan68.jpg', 'vartan69.jpeg', 'vartan7.jpg', 
'vartan70.jpg', 'vartan71.jpg', 'vartan72.jpg', 'vartan73.jpg', 'vartan74.jpg', 'vartan75.jpg', 'vartan76.jpg', 
'vartan77.jpg', 'vartan78.jpg', 'vartan79.jpg', 'vartan8.jpg', 'vartan80.jpg', 'vartan81.jpg', 'vartan82.jpg', 
'vartan83.jpg', 'vartan84.jpg', 'vartan85.jpg', 'vartan86.jpg', 'vartan87.jpg', 'vartan88.jpg', 'vartan89.jpg', 
'vartan9.jpg' ]
    if not os.path.exists('training_set'):
        os.makedirs('training_set')
    for f in fileToMove:
        shutil.move('cropped/'+f, 'training_set/'+f)

def OrganizeValidationSet():
    fileToMove = ['bracco18.jpg', 'bracco27.jpg', 'bracco30.jpg', 'bracco32.jpg', 'bracco33.jpg', 'bracco34.jpg', 'bracco35.jpg', 'bracco36.jpg', 'bracco47.jpg', 'bracco67.jpg', 'butler100.jpg', 'butler91.jpg', 'butler92.jpeg', 'butler93.jpg', 'butler94.jpg', 'butler95.jpg', 'butler96.jpg', 'butler97.jpg', 'butler98.jpg', 'butler99.jpg', 'gilpin100.jpg', 'gilpin101.jpg', 'gilpin92.jpg', 'gilpin93.jpg', 'gilpin94.jpg', 'gilpin95.jpg', 'gilpin96.jpg', 'gilpin97.png', 'gilpin98.jpg', 'gilpin99.jpg', 'harmon100.jpg', 'harmon101.jpg', 'harmon102.jpg', 'harmon103.jpg', 'harmon104.jpg', 'harmon105.jpg', 'harmon106.jpg', 'harmon97.jpg', 'harmon98.jpg', 'harmon99.jpg', 'radcliffe18.jpg', 'radcliffe22.jpg', 'radcliffe23.jpg', 'radcliffe24.jpg', 'radcliffe25.jpg', 'radcliffe32.jpg', 'radcliffe33.jpg', 'radcliffe34.jpg', 'radcliffe35.jpg', 'radcliffe45.jpg', 'vartan90.jpg', 'vartan91.jpg', 'vartan92.jpg', 'vartan93.jpg', 'vartan94.jpg', 'vartan95.jpg', 'vartan96.jpg', 'vartan97.jpg', 'vartan98.jpg', 'vartan99.jpg']
    if not os.path.exists('validation_set'):
        os.makedirs('validation_set')
    for f in fileToMove:
        shutil.move('cropped/'+f, 'validation_set/'+f)
        
def OrganizeTestSet():
    fileToMove = ['bracco129.jpg', 'bracco130.jpg', 'bracco131.jpg', 'bracco132.jpg', 'bracco139.jpg', 'bracco140.jpg', 'bracco141.jpg', 'bracco143.jpg', 'bracco145.jpg', 'bracco146.jpg', 'butler142.jpg', 'butler143.jpg', 'butler144.jpg', 'butler145.jpg', 'butler147.jpg', 'butler148.jpg', 'butler149.jpg', 'butler150.jpg', 'butler151.jpg', 'butler152.jpg', 'gilpin131.jpg', 'gilpin134.jpg', 'gilpin135.jpg', 'gilpin136.jpg', 'gilpin142.jpg', 'gilpin143.jpg', 'gilpin144.jpg', 'gilpin145.jpg', 'gilpin146.jpg', 'gilpin147.jpg', 'harmon151.png', 'harmon152.png', 'harmon153.jpg', 'harmon154.jpg', 'harmon155.jpg', 'harmon156.jpg', 'harmon157.jpg', 'harmon158.jpg', 'harmon159.jpg', 'harmon160.jpg', 'radcliffe124.jpg', 'radcliffe125.jpg', 'radcliffe132.jpg', 'radcliffe134.jpg', 'radcliffe135.jpg', 'radcliffe136.jpg', 'radcliffe137.jpg', 'radcliffe138.jpg', 'radcliffe139.jpg', 'radcliffe140.jpg', 'vartan135.jpg', 'vartan136.jpg', 'vartan137.jpg', 'vartan138.jpg', 'vartan139.jpg', 'vartan140.jpg', 'vartan141.jpg', 'vartan142.jpg', 'vartan143.jpg', 'vartan144.jpg']
    if not os.path.exists('test_set'):
        os.makedirs('test_set')
    for f in fileToMove:
        shutil.move('cropped/'+f, 'test_set/'+f)
        
def OrganizeTestSet2():
    fileToMove = ['anderson110.jpg', 'anderson111.jpg','anderson112.jpg','anderson113.jpg','anderson114.jpg','anderson116.jpg','anderson117.jpg','anderson118.jpg','anderson119.jpg','anderson120.jpg','cattrall46.jpg', 'cattrall47.jpg', 'cattrall48.jpg', 'cattrall49.jpg','cattrall50.jpg','cattrall51.jpg','cattrall52.jpg','cattrall53.jpg','cattrall54.jpg','cattrall55.jpg','conn52.jpg',
'conn53.jpg','conn56.gif','conn57.jpg','conn58.jpg','conn60.jpg','conn61.jpg','conn62.jpg','conn64.jpg','conn65.JPG','delany18.jpg', 'delany19.jpg','delany20.jpg','delany21.jpg','delany22.jpg','delany23.jpg','delany24.jpg','delany25.jpg','delany26.jpg','delany27.jpg','dicaprio14.jpg',
'dicaprio15.jpg','dicaprio16.jpg','dicaprio17.jpg','dicaprio18.jpg','dicaprio19.jpg','dicaprio20.jpg','dicaprio21.jpg','dicaprio22.jpg','dicaprio23.jpg','dourdan17.jpg',
'dourdan18.jpg','dourdan19.jpg','dourdan20.jpg','dourdan22.jpg','dourdan23.jpg','dourdan24.jpg','dourdan25.jpg','dourdan26.jpg','dourdan27.jpg','electra10.jpg',
'electra11.jpg','electra12.jpg','electra13.jpg','electra15.jpg','electra16.jpg','electra17.jpg','electra18.jpg','electra19.jpg','electra20.jpg','elwes0.jpg',
'elwes1.jpg','elwes2.jpg','elwes3.jpg','elwes4.jpg','elwes5.jpeg','elwes6.jpg','elwes7.jpg','elwes8.jpg','elwes9.jpg','hartley12.jpg',
'hartley13.jpg','hartley15.jpg','hartley16.jpg','hartley17.jpg','hartley18.png','hartley19.jpg','hartley20.jpg','hartley21.jpg','hartley22.jpg','innes11.jpg',
'innes12.jpg','innes13.jpg','innes14.jpg','innes15.jpg','innes16.jpg','innes17.jpeg','innes18.jpg','innes19.jpg','innes20.jpg','klein10.jpg',
'klein11.jpg','klein12.jpg','klein13.jpg','klein14.jpg','klein15.jpg','klein16.jpg','klein17.jpg','klein18.jpg','klein19.jpg','long33.jpg',
'long34.jpg','long35.jpg','long36.jpg','long37.jpg','long38.png','long39.jpeg','long40.jpg','long41.jpg','long42.jpg','louis-dreyfus32.jpg',
'louis-dreyfus33.jpg','louis-dreyfus35.jpg','louis-dreyfus36.jpg','louis-dreyfus37.jpg','louis-dreyfus38.jpg','louis-dreyfus39.jpg','louis-dreyfus40.jpg','louis-dreyfus41.jpg','louis-dreyfus42.jpg','madden11.jpg',
'madden12.jpg','madden13.jpg','madden14.jpg','madden16.jpg','madden17.jpg','madden18.jpg','madden19.jpg','madden20.jpg','madden21.jpg','marcil0.jpg',
'marcil1.jpg','marcil2.jpg','marcil3.jpg','marcil4.jpg','marcil5.jpg','marcil6.jpg','marcil7.jpg','marcil8.jpg','marcil9.jpg','meyer55.jpg',
'meyer56.jpg','meyer57.jpg','meyer58.jpg','meyer59.jpg','meyer60.jpg','meyer61.jpg','meyer62.jpg','meyer63.jpg','meyer64.jpg','noth11.jpg',
'noth12.jpg','noth13.jpg','noth14.jpg','noth16.jpg','noth17.jpg','noth18.jpg','noth19.jpg','noth20.jpg','noth22.jpg','richter10.jpg',
'richter11.jpg','richter2.jpg','richter3.jpg','richter4.jpg','richter5.jpg','richter6.jpg','richter7.jpg','richter8.jpg','richter9.jpg','smith21.jpg',
'smith22.jpg','smith23.jpg','smith24.jpg','smith25.jpg','smith26.jpg','smith27.jpg','smith28.jpg','smith29.jpg','smith30.jpg','statham26.jpg',
'statham27.jpg','statham28.jpg','statham29.jpg','statham30.jpg','statham31.jpg','statham32.jpg','statham33.jpg','statham34.jpg','statham35.jpg','walker13.jpg',
'walker14.jpg','walker15.jpg','walker16.jpg','walker17.jpg','walker18.jpg','walker19.jpg','walker20.jpg','walker21.jpg','walker22.jpg']
    if not os.path.exists('test_set2'):
        os.makedirs('test_set2')
    for f in fileToMove:
        shutil.move('cropped/'+f, 'test_set2/'+f)

def OrganizeValidationSet2():
    fileToMove = ['anderson100.jpg', 'anderson101.jpg', 'anderson102.jpg', 'anderson103.jpg', 'anderson104.jpg', 'anderson105.jpg', 'anderson106.jpg', 'anderson107.jpg', 'anderson108.jpg', 'anderson109.jpg', 'cattrall1.jpg', 'cattrall10.jpg', 'cattrall2.jpg', 'cattrall3.jpg', 'cattrall4.jpg', 'cattrall5.jpg', 'cattrall6.jpg', 'cattrall7.jpg', 'cattrall8.jpg', 'cattrall9.jpg', 'conn10.jpg', 'conn11.jpg', 'conn12.jpg', 'conn13.jpg', 'conn14.jpg', 'conn16.jpg', 'conn6.jpg', 'conn7.jpg', 'conn8.jpg', 'conn9.jpg', 'delany10.jpg', 'delany11.jpg', 'delany12.jpg', 'delany13.jpg', 'delany14.jpg', 'delany15.jpg', 'delany16.jpg', 'delany17.jpg', 'delany8.jpg', 'delany9.jpg', 'dicaprio26.png', 'dicaprio27.jpeg', 'dicaprio28.jpg', 'dicaprio29.jpg', 'dicaprio30.jpg', 'dicaprio31.jpg', 'dicaprio32.jpg', 'dicaprio33.jpg', 'dicaprio35.jpeg', 'dicaprio37.jpg', 'dourdan0.jpg', 'dourdan1.jpg', 'dourdan2.jpg', 'dourdan3.jpg', 'dourdan4.jpg', 'dourdan5.jpg', 'dourdan6.jpg', 'dourdan7.jpg', 'dourdan8.jpg', 'electra22.jpg', 'electra24.jpg', 'electra26.jpg', 'electra27.jpg', 'electra29.jpg', 'electra30.jpg', 'electra31.jpg', 'electra32.jpg', 'electra33.jpg', 'electra34.jpg', 'elwes10.jpg', 'elwes11.jpg', 'elwes12.jpg', 'elwes13.jpg', 'elwes14.jpg', 'elwes15.jpg', 'elwes16.jpg', 'elwes17.jpg', 'elwes18.jpg', 'elwes19.jpg', 'hartley0.jpg', 'hartley1.jpg', 'hartley2.jpg', 'hartley25.jpg', 'hartley3.jpg', 'hartley4.jpg', 'hartley5.jpg', 'hartley6.jpg', 'hartley7.jpg', 'hartley8.jpg', 'hartley9.jpg', 'innes1.jpg', 'innes10.jpg', 'innes2.jpg', 'innes3.jpg', 'innes4.jpg', 'innes5.jpg', 'innes6.jpg', 'innes7.jpeg', 'innes8.jpg', 'innes9.jpeg', 'klein0.jpg', 'klein1.jpg', 'klein2.jpg', 'klein3.jpg', 'klein4.jpg', 'klein5.jpg', 'klein6.jpg', 'klein7.jpg', 'klein8.jpg', 'klein9.jpg', 'long22.jpg', 'long23.jpg', 'long24.jpg', 'long25.jpg', 'long26.jpg', 'long27.jpg', 'long28.jpg', 'long29.jpg', 'long31.jpg', 'long32.jpg', 'louis-dreyfus21.jpg', 'louis-dreyfus22.jpg', 'louis-dreyfus23.jpg', 'louis-dreyfus24.jpg', 'louis-dreyfus25.jpg', 'louis-dreyfus26.jpg', 'louis-dreyfus27.jpg', 'louis-dreyfus28.jpg', 'louis-dreyfus29.jpg', 'louis-dreyfus30.jpg', 'madden0.png', 'madden1.jpg', 'madden2.jpg', 'madden3.jpg', 'madden4.jpg', 'madden5.jpg', 'madden6.jpg', 'madden7.jpg', 'madden8.jpg', 'madden9.jpg', 'marcil10.jpg', 'marcil11.jpg', 'marcil12.jpg', 'marcil14.jpg', 'marcil15.jpg', 'marcil16.jpg', 'marcil17.jpg', 'marcil18.jpg', 'marcil20.jpg', 'marcil24.jpg', 'meyer40.jpg', 'meyer41.jpg', 'meyer42.jpg', 'meyer43.jpg', 'meyer44.jpg', 'meyer45.jpg', 'meyer46.jpg', 'meyer47.jpg', 'meyer48.jpg', 'meyer49.jpg', 'noth0.jpg', 'noth1.jpg', 'noth2.jpg', 'noth3.jpg', 'noth4.jpg', 'noth5.jpg', 'noth6.jpg', 'noth7.jpg', 'noth8.jpg', 'noth9.jpg', 'richter12.jpg', 'richter13.jpg', 'richter14.jpg', 'richter15.jpg', 'richter16.jpg', 'richter17.jpg', 'richter18.jpg', 'richter19.jpg', 'richter20.jpg', 'richter21.jpg', 'smith31.jpg', 'smith32.jpg', 'smith33.jpg', 'smith34.jpg', 'smith35.jpg', 'smith36.jpg', 'smith37.jpg', 'smith38.jpg', 'smith39.jpg', 'smith40.jpg', 'statham16.jpg', 'statham17.jpg', 'statham18.jpg', 'statham19.jpg', 'statham20.jpg', 'statham21.jpg', 'statham22.jpg', 'statham23.jpg', 'statham24.jpg', 'statham25.jpg', 'walker1.jpg', 'walker10.jpg', 'walker11.jpg', 'walker12.jpg', 'walker2.jpg', 'walker3.jpg', 'walker4.jpg', 'walker5.jpg', 'walker7.jpg', 'walker8.jpg' ]
    if not os.path.exists('validation_set2'):
        os.makedirs('validation_set2')
    for f in fileToMove:
        shutil.move('cropped/'+f, 'validation_set2/'+f)

def kNN(k, lblimg, trainingSet):
    distanceList = []
    fiveNearestNeighbours = []
    
    actRanking = [ LabeledNumber(0, 'butler'), LabeledNumber(0, 'radcliffe'), LabeledNumber(0, 'vartan'), LabeledNumber(0, 'bracco'), LabeledNumber(0, 'gilpin'), LabeledNumber(0, 'harmon')]
    
    for index in range(0, len(trainingSet)):
        sum = 0
        for i in range (0, 32):
            for j in range(0, 32):
                distance = float(lblimg.img[i, j])-float(trainingSet[index].img[i, j])
                distance = distance * distance
                sum += distance
        distanceList.append(LabeledNumber(math.sqrt(sum), trainingSet[index].name))
        fiveNearestNeighbours.append(LabeledNumber(math.sqrt(sum), trainingSet[index].filename))
    distanceList = sorted(distanceList, key=lambda distance: distance.number)
    fiveNearestNeighbours = sorted(fiveNearestNeighbours, key=lambda distance: distance.number)
    
    for i in range(0,k):
        if distanceList[i].name == 'butler':
            actRanking[0].number += 1
        elif distanceList[i].name == 'radcliffe':
            actRanking[1].number += 1
        elif distanceList[i].name == 'vartan':
            actRanking[2].number += 1
        elif distanceList[i].name == 'bracco':
            actRanking[3].number += 1
        elif distanceList[i].name == 'gilpin':
            actRanking[4].number += 1
        elif distanceList[i].name == 'harmon':
            actRanking[5].number += 1
    print '\nK = '+str(k)
    actRanking = sorted(actRanking, key=lambda act: act.number)
    for a in actRanking:
        print a.name, a.number
    
   
    print 'Filename: ' + lblimg.filename + ' Name: ' + lblimg.name +' Associated with: ' + actRanking[5].name
    print 'Filename of Five Nearest Neighbours: '
    str_ = ''
    #For the testing 
    for i in range (0, 5):
        str_ = str_ +' '+ str(i+1) + ' - ' + fiveNearestNeighbours[i].name +'\n'
    print str_
    
    if lblimg.name == actRanking[5].name:
        return True
    return False
   
def kNN_gendered(k, lblimg, trainingSet):
    distanceList = []
    fiveNearestNeighbours = []
    
    actRanking = [ LabeledNumber(0, 'male'), LabeledNumber(0, 'female')]
    
    for index in range(0, len(trainingSet)):
        sum = 0
        for i in range (0, 32):
            for j in range(0, 32):
                distance = float (lblimg.img[i, j])-float(trainingSet[index].img[i, j])
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

#Running the code

if sys.argv[1] == '-org':                
    #Organizing the folders
    OrganizeTrainingSet()
    OrganizeValidationSet()
    OrganizeTestSet()
    
    OrganizeTestSet2()

#Face Recognition 

#1 - Loading the training set
act = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

for i in range (0, len(act)):
    act[i] = act[i].split()[1].lower()

filelist = os.listdir('training_set')

trainingSet = []
validatedSet = []


for i in range(0, len(filelist)):
    for j in range (0, len(act)):
        if act[j] in filelist[i]:
            trainingSet.append(LabeledImg(act[j], scipy.misc.imread('training_set/'+filelist[i]),filelist[i] )) #append a Labeled obj
            break

if sys.argv[1] == '-face' and sys.argv[2] == '-val':
    #2 - Loading the validation set
    filelist2 = os.listdir('validation_set')
    
    correct = 0
    total = 0
    
    butler = 0
    radc = 0
    bracco = 0
    vartan = 0
    harmon = 0
    gilpin = 0
    
    #2.1 - Creating kResults list to generate plot
    kResults = []
    for k in range(1, 15):
        tempNbr = LabeledNumber(0, str(k) )
        kResults.append(tempNbr)
    
    #2.2 - Checking accuracy for possible Ks
    for i in range(0, len(filelist2)):
        #To each image in validation_set directory
        total += 1
        for j in range (0, len(act)):
            if act[j] in filelist2[i]:
                temp_img = scipy.misc.imread('validation_set/'+filelist2[i])
                temp_labeledObj = LabeledImg(act[j], temp_img, filelist2[i])
                validatedSet.append(temp_labeledObj) #append a Labeled obj
                
                for k in range(1, 15):
                    #calculate all the distances
                    if kNN(k, temp_labeledObj, trainingSet):
                        kResults[k-1].number += 1
                        if temp_labeledObj.name == 'butler':
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
                            vartan += 1
                break
    
    x = []
    y = []
    for k in range(0, 14):
        temp = float(kResults[k].number)/float(60)
        x.append(k+1)
        y.append(temp)
        print str(k) + ': '+str(temp)
    plt.plot(x, y, 'bo')
    plt.title('Accuracy of kNN algorithm, using different values of k, for \nface recognition in pictures from validation set ')
    plt.xlabel('K values')
    plt.ylabel('Accuracy (%)')  
    plt.ylim([0,1])
    plt.xlim([0,15])
    plt.savefig('Accuracy_K_plot1.png')
    plt.close()
    
    print "Graph saved as 'Accuracy_K_plot1.png' in the current folder"
    plt.clf()

#3 - Using test_set with k = 1 
if sys.argv[1] == '-face' and sys.argv[2] == '-test':
    testSet = []
    filelist3 = os.listdir('test_set')
    
    kResults = []
    for k in range(1, 15):
        tempNbr = LabeledNumber(0, str(k) )
        kResults.append(tempNbr)
    
    correct = 0
    total = 0
    for i in range(0, len(filelist3)):
        #load the images
        total += 1
        for j in range (0, len(act)):
            if act[j] in filelist3[i]:
                temp_img = scipy.misc.imread('test_set/'+filelist3[i])
                temp_labeledObj = LabeledImg(act[j], temp_img, filelist3[i])
                testSet.append(temp_labeledObj) #append a Labeled obj
                if len(sys.argv) > 3 and sys.argv[3] == '-k':
                    for k in range(1, 15):
                        #calculate all the distances
                        if kNN(k, temp_labeledObj, trainingSet): 
                            kResults[k-1].number += 1
                else:
                    k=1
                    if kNN(k, temp_labeledObj, trainingSet):
                        correct += 1
    if len(sys.argv) > 3 and sys.argv[3] == '-k':
        x = []
        y = []
        for k in range(0, 14):
            temp = float(kResults[k].number)/float(60)
            x.append(k+1)
            y.append(temp)
            print str(k) + ': '+str(temp)
        plt.plot(x, y, 'bo')
        plt.title('Accuracy of kNN algorithm, using different values of k, for \nface recognition in pictures from test set ')
        plt.xlabel('K values')
        plt.ylabel('Accuracy (%)')  
        plt.ylim([0,1])
        plt.xlim([0,15])
        plt.savefig('Accuracy_K_plot_testset.png')
        plt.close()
    else:
        print 'Accuracy of the face recognition such that k = 1:\n'
        print float(correct)/float(total)

#4 - Gender recognition using the validation set
if sys.argv[1] == '-gender' and sys.argv[2] == '-val':

    print '\n\n ******* Gender Recognition using validation set **********\n\n'
    print 'Defyning K'
    
    filelist2 = os.listdir('validation_set')
    kResults = []
    
    for k in range(1, 15):
        tempNbr = LabeledNumber(0, str(k) )
        kResults.append(tempNbr)
    
    total = 0
    for i in range(0, len(filelist2)):
        #load the images
        total += 1
        for j in range (0, len(act)):
            if act[j] in filelist2[i]:
                temp_img = scipy.misc.imread('validation_set/'+filelist2[i])
                if j < 3:
                    temp_labeledObj = LabeledImg(act[j], temp_img, filelist2[i], 'male')
                else:
                    temp_labeledObj = LabeledImg(act[j], temp_img, filelist2[i], 'female')
                validatedSet.append(temp_labeledObj) #append a Labeled obj
                
                for k in range(1, 15):
                    #calculate all the distances
                    if kNN_gendered(k, temp_labeledObj, trainingSet):
                        kResults[k-1].number += 1
    
    x = []
    y = []
    for k in range(0, 14):
        temp = kResults[k].number/float(60)
        x.append(k+1)
        y.append(temp)
        print 'k = ' + str(kResults[k].number) + ' Perc: ' + str(temp)
    plt.plot(x, y, 'bo')
    plt.title('Accuracy of the gender recognition using kNN algorithm\napplied in the validation set')
    plt.xlabel('K values')
    plt.ylabel('Accuracy (%)')  
    plt.ylim([0,1])
    plt.xlim([0,15])
    plt.savefig('Accuracy_K_gender_val.png')
    plt.close()
    plt.clf()
    
    print "The plot was saved in the current folder as 'Accuracy_K_gender_val.png'"


#Gender recognition of the actors in the test set
if sys.argv[1] == '-gender' and sys.argv[2] == '-test':
    print '\n\n ******* Gender Recognition using test set **********\n\n'
    print 'K defined: k = 1'
    filelist3 = os.listdir('test_set')
    kResults = []
    
    for k in range(1, 15):
        tempNbr = LabeledNumber(0, str(k) )
        kResults.append(tempNbr)
    
    correct = 0
    total = 0
    for i in range(0, len(filelist3)):
        #load the images
        total += 1
        for j in range (0, len(act)):
            if act[j] in filelist3[i]:
                temp_img = scipy.misc.imread('test_set/'+filelist3[i])
                if j < 3:
                    temp_labeledObj = LabeledImg(act[j], temp_img, filelist3[i], 'male')
                else:
                    temp_labeledObj = LabeledImg(act[j], temp_img, filelist3[i],'female')
                validatedSet.append(temp_labeledObj) #append a Labeled obj
                k=1
                if len(sys.argv) > 3 and sys.argv[3] == '-k':
                    for k in range(1, 15):
                        if kNN_gendered(k, temp_labeledObj, trainingSet):
                            kResults[k-1].number += 1
                elif kNN_gendered(k, temp_labeledObj, trainingSet):
                        correct += 1
    if len(sys.argv) > 3 and sys.argv[3] == '-k':
        x = []
        y = []
        for k in range(0, 14):
            temp = float(kResults[k].number)/float(60)
            x.append(k+1)
            y.append(temp)
            print str(k) + ': '+str(temp)
        plt.plot(x, y, 'bo')
        plt.title('Accuracy of kNN algorithm, using different values of k, for \ngender recognition in pictures from test set ')
        plt.xlabel('K values')
        plt.ylabel('Accuracy (%)')  
        plt.ylim([0,1])
        plt.xlim([0,15])
        plt.savefig('Accuracy_K_plot_testset_gender.png')
        plt.close()
    else:
        acc = float(correct)/float(total)
        print "Accuracy = "+str(acc)


#Part 6 - Gender recognition of other actors using the current training set
if sys.argv[1] == '-new' and sys.argv[2] == '-val':
    input_files = ['subset_actresses.txt', 'subset_actors.txt']
    
    act = list(set([a.split("\t")[0] for a in open("subset_actresses.txt").readlines()]))
    act = act + list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()])) #concatenate lists
    
    for i in range(0, len(act)):
        act[i] = act[i].split()[-1].lower()
        
    filename_cropped = os.listdir('validation_set2')
    croppedImg = []
    kResults = []
    for k in range(1, 15):
        tempNbr = LabeledNumber(0, str(k) )
        kResults.append(tempNbr)
    
    correct = 0
    total = 0
    for f in filename_cropped:
        total += 1
        for i in range(0, len(act)):
            if act[i] in f:
                temp_img = scipy.misc.imread('validation_set2/'+f)
                if i < 15:
                    print act[i] + ' ' + 'female'
                    temp_labeledObj = LabeledImg(act[i], temp_img, f, 'female')
                    croppedImg.append(temp_labeledObj)
                else:
                    print act[i] + ' ' + 'male'
                    temp_labeledObj = LabeledImg(act[i], temp_img, f, 'male')
                    croppedImg.append(temp_labeledObj)
                for k in range(1,15):
                    if kNN_gendered(k, temp_labeledObj, trainingSet):
                            kResults[k-1].number += 1
    x = []
    y = []
    for k in range(0, 14):
        temp = float(kResults[k].number)/float(total)
        x.append(k+1)
        y.append(temp)
        print str(k) + ': '+str(temp)
    plt.plot(x, y, 'bo')
    plt.title('Accuracy of kNN algorithm, using different values of k, for \ngender recognition in pictures from the new validation set')
    plt.xlabel('K values')
    plt.ylabel('Accuracy (%)')  
    plt.ylim([0,1])
    plt.xlim([0,15])
    plt.savefig('Accuracy_K_plot_newvalset_gender.png')
    plt.close()


if sys.argv[1] == '-new' and sys.argv[2] == '-test':
    print '\n\n ******* Gender Recognition using test set **********\n\n'
    print 'K defined: k = 3'
    
    input_files = ['subset_actresses.txt', 'subset_actors.txt']
    
    act = list(set([a.split("\t")[0] for a in open("subset_actresses.txt").readlines()]))
    act = act + list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()])) #concatenate lists
    
    for i in range(0, len(act)):
        act[i] = act[i].split()[-1].lower()
    
    filelist3 = os.listdir('test_set2')
    kResults = []
    
    for k in range(1, 15):
        tempNbr = LabeledNumber(0, str(k) )
        kResults.append(tempNbr)
    
    correct = 0
    total = 0
    for i in range(0, len(filelist3)):
        #load the images
        total += 1
        for j in range (0, len(act)):
            if act[j] in filelist3[i]:
                temp_img = scipy.misc.imread('test_set2/'+filelist3[i])
                if j < 15:
                    temp_labeledObj = LabeledImg(act[j], temp_img, filelist3[i], 'female')
                else:
                    temp_labeledObj = LabeledImg(act[j], temp_img, filelist3[i],'male')
                validatedSet.append(temp_labeledObj) #append a Labeled obj
                k=3
                if len(sys.argv) > 3 and sys.argv[3] == '-k':
                    for k in range(1, 15):
                        if kNN_gendered(k, temp_labeledObj, trainingSet):
                            kResults[k-1].number += 1
                elif kNN_gendered(k, temp_labeledObj, trainingSet):
                        correct += 1
    if len(sys.argv) > 3 and sys.argv[3] == '-k':
        x = []
        y = []
        for k in range(0, 14):
            temp = float(kResults[k].number)/float(total)
            x.append(k+1)
            y.append(temp)
            print str(k) + ': '+str(temp)
        plt.plot(x, y, 'bo')
        plt.title('Accuracy of kNN algorithm, using different values of k, for \ngender recognition in pictures from test set ')
        plt.xlabel('K values')
        plt.ylabel('Accuracy (%)')  
        plt.ylim([0,1])
        plt.xlim([0,15])
        plt.savefig('Accuracy_K_plot_testset2_gender.png')
        plt.close()
    else:
        acc = float(correct)/float(total)
        print "Accuracy = "+str(acc)