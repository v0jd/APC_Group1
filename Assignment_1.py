import cv2 as cv
import numpy as np

Color=(204,204,204)

img = np.zeros((512,512,3), np.uint8)
cv.circle(img,(256,256),240,Color,10)

Center_triangle=np.array([[206,236],[306,236],[256,306]],np.int32)
Center_triangle=Center_triangle.reshape((-1,1,2))
cv.fillPoly(img,[Center_triangle],Color)

UpperTriangle=np.array([[206,236],[306,236],[256,26]],np.int32)
UpperTriangle=UpperTriangle.reshape((-1,1,2))
cv.fillPoly(img,[UpperTriangle],Color)

LeftTriangle=np.array([[206,236],[256,306],[36,366]],np.int32)
LeftTriangle=LeftTriangle.reshape((-1,1,2))
cv.fillPoly(img,[LeftTriangle],Color)

RightTriangle=np.array([[306,236],[256,306],[475,366]],np.int32)
RightTriangle=RightTriangle.reshape((-1,1,2))
cv.fillPoly(img,[RightTriangle],Color)

cv.imshow("display",img)

k=cv.waitKey(0)

