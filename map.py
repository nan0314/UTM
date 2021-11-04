#!/usr/bin/env python3

import cv2
import numpy as np

# read image in as numpy array
img = cv2.imread('LA.png')   # reads an image in the BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB

# set up risk map

risk_map = np.zeros((img.shape[0],img.shape[1]))
b = 1.5 # risk constant

# initialize color bounds

boundaries = [ ([235, 220, 100], [255, 240, 220], b**-3),      # brown - desert + roads
               ([130, 160, 225], [165, 210, 255], b**-2),      # blue - water
               ([10,150,10], [220,255,220], b**-1),            # green - mountain
               ([220, 220, 220], [250, 250, 250], b**0),       # grey - city
               ([240,100,150], [255,250,200], b**0),           # orange - roads
               ([190,50,20], [255,255,170], b**0) ]            # orange/brown - road borders

for (lower, upper, risk) in boundaries:
    mask = cv2.inRange(img,np.array(lower),np.array(upper))/255*risk
    risk_map = np.maximum(risk_map,mask)

other_mask = (risk_map < 1e-5)*b**0
risk_map = np.maximum(risk_map,other_mask)

cv2.imwrite('risk_map.jpg',risk_map*255)
print(risk_map.shape)

