import skimage.data as skid
import cv2
import pylab as plt
import scipy.misc


img = scipy.misc.face()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20,10))
plt.imshow(img)
plt.show()

sift = cv2.xfeatures2d.SIFT_create()

step_size = 5
kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                                    for x in range(0, gray.shape[1], step_size)]

img=cv2.drawKeypoints(gray,kp, img)

plt.figure(figsize=(20,10))
plt.imshow(img)
plt.show()

dense_feat = sift.compute(gray, kp)