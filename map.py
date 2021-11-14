#!/usr/bin/env python3

import cv2
import numpy as np

#################################
# Helper Functions
#################################

def circle_obstacle(x,y,r,map,num=100):
    xx,yy = np.mgrid[-x:map.shape[0]-x, -y:map.shape[1]-y]
    mask = xx*xx + yy*yy <= r*r
    map[mask] = num


#################################
# Risk Map Generation
#################################

# read image in as numpy array
img = cv2.imread('images/LA.png')   # reads an image in the BGR format
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

kernel = np.ones((6,6), np.uint8)

for (lower, upper, risk) in boundaries:
    mask = cv2.inRange(img,np.array(lower),np.array(upper))/255*risk # find specific color (see boundaries)
    closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # perform closing operation to remove noise
    risk_map = np.maximum(risk_map,closed) # save the higher risk value

other_mask = (risk_map < 1e-5)*b**0        # untouched pixels
risk_map = np.maximum(risk_map,other_mask) # save the higher risk value

# Add airport no fly regions
circle_obstacle(670,420,30,risk_map,b**2)   # LAX
circle_obstacle(900,775,30,risk_map,b**2)   # John Wayne
circle_obstacle(1100, 420,3,risk_map,b**2)

# np.savetxt("risk_map.csv", risk_map, delimiter=",")
cv2.imwrite('images/risk_map.jpg',risk_map*255/np.max(risk_map))

dl = 58.57/((670 - 900)**2 + (420 - 775)**2)**0.5  # calculate pixel size in km based off distance between LAX and John Wayne


