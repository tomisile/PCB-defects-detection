import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils

test = cv.imread('voc4.jpg') # queryImage
template = cv.imread('voc3.jpg') # Reference Image


################# REGISTRATION ##################

e1 = cv.getTickCount()

# convert to grayscale
img1 = cv.cvtColor(test, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv.SIFT_create()

'''# draw keypoints only 
kepy1 = sift.detect(img1,None)
kepy2 = sift.detect(img2,None)
img1kp=cv.drawKeypoints(img1,kepy1,img1)
img2kp=cv.drawKeypoints(img2, kepy2, img2)
#plt.imshow(img1kp, 'gray'), plt.show()
#plt.imshow(img2kp, 'gray'), plt.show()'''

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

print('debug: len(good) = ', len(good))

MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
#plt.imshow(img2, 'gray'), plt.show()

'''draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#plt.imshow(img3, 'gray'),plt.show()'''

# Use homography
height, width = img2.shape[:2]
imgtrans = cv.warpPerspective(img1, M, (width, height))
#plt.imshow(imgtrans, 'gray'), plt.show()

e2 = cv.getTickCount()
regtime = (e2 - e1)/cv.getTickFrequency()
print('Registration time:', regtime, 'seconds')


################ BINARIZATION ################

e3 = cv.getTickCount()

# Using adaptive gaussian thresholding
templatebinary = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv.THRESH_BINARY,11,2)
testbinary = cv.adaptiveThreshold(imgtrans,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv.THRESH_BINARY,11,2)
#plt.imshow(templatebinary, 'gray'), plt.show()
#plt.imshow(testbinary, 'gray'), plt.show()

e4 = cv.getTickCount()
bintime = (e4 - e3)/cv.getTickFrequency()
print('Binarization time:', bintime, 'seconds')


############# LOCALIZATION ###############

e5 = cv.getTickCount()

# xor
imgxor = cv.bitwise_xor(templatebinary, testbinary)
#plt.imshow(imgxor, 'gray'), plt.show()

# median filtering 5x5
imgmed1 = cv.medianBlur(imgxor, 5)

# closing 15x15(rectangle)
kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
closing1 = cv.morphologyEx(imgmed1, cv.MORPH_CLOSE, kernel1)

# opening 3x3(rectangle)
kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
opening1 = cv.morphologyEx(closing1, cv.MORPH_OPEN, kernel2)

# median filtering 5x5
imgmed2 = cv.medianBlur(opening1, 5)

# closing 29x29(ellipse)
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(29,29))
closing2 = cv.morphologyEx(imgmed1, cv.MORPH_CLOSE, kernel3)

# opening 3x3(rectangle)
opening2 = cv.morphologyEx(closing1, cv.MORPH_OPEN, kernel2)

# opening 1x1(rectangle)
kernel4 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
opening3 = cv.morphologyEx(closing1, cv.MORPH_OPEN, kernel4)

#plt.imshow(opening3, 'gray'), plt.show()

e6 = cv.getTickCount()
loctime = (e6 - e5)/cv.getTickFrequency()
print('Localization time:', loctime, 'seconds')


# detect edges using canny
edges = cv.Canny(image=opening3, threshold1=30, threshold2=200)


# find contours
contours = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

new = imgtrans.copy()

X=[]
Y=[]
CX=[]
CY=[]
C=[]

for c in contours:
	M = cv.moments(c)
	if(M["m00"] != 0):
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])
		CX.append(cx)
		CY.append(cy)
		C.append((cx,cy))

print(CX)
print(CY)

implot = plt.imshow(new)
plt.scatter(CX , CY , c='r' , s=40)
plt.show()

'''for c in contours:
    # fit a bounding box to the contour
    (x, y, w, h) = cv.boundingRect(c)
    cv.rectangle(new, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
cv.imshow('local. defect', new)
cv.waitKey()'''
