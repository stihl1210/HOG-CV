import cv2
import numpy as np

img = cv2.imread("gradient.png");
img = np.float32(img) / 255.0

width_old = img.shape[0]
height_old = img.shape[1]

print(img.shape)
w = width_old%8
h = height_old%8


w_start = int(w/2)

w_stop = width_old - w_start

h_start = int(h/2)

h_stop = height_old - h_start
print(w_start, w_stop, h_start, h_stop)

img= img[h_start:h_stop, w_start:w_stop]


dividerx = img.shape[0]/128
dividery = img.shape[1]/128

img = cv2.resize(img, ( int(img.shape[0]/dividerx), int(img.shape[0]/dividery)), interpolation=cv2.INTER_LINEAR )

print(img.shape)

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 1);
gy = cv2.Sobel(img,  cv2.CV_32F, 0, 1, 1);

print(gx.shape)


mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

print(mag.shape)


hist = np.zeros((int(mag.shape[0]/8), int(mag.shape[1]/8), 9))


def getAngles(ang):
    ang = ang%180
    trueang = ang
    if(ang <= 180 and ang>=160):
        ang1 = 160
        ang2 = 0
        l = trueang - 160
        return (ang, int(ang1/20), int(ang2/20), 1- l/20, l/20 )
    else:

        trueang = ang
        ang = ang%160

        trueang2 = ang

        val = ang - ang%20

        mod = val + 20

        l = trueang2-val
        p = mod - trueang2

        if(l+p !=0):
            return trueang, int(val/20), int(mod/20), p/(l+p) , l/(l+p)
        else:
            l = 0.5
            p = 0.5
            return val, mod, l , p

# for i in range(360):
#     print(getAngles(i))

for px in range(int(mag.shape[0]/8)):
    for py in range(int(mag.shape[1]/8)):

        xs = px * 8
        ys = py * 8
        for x in range(8):
            for y in range (8):
                val = mag[xs + x ,ys + y][0];
                ang = angle[int(xs + x) ,int(ys + y) ][0];

                val, id1, id2, fp, fp2 = getAngles(int(ang));

                hist[px,py,id1]+= fp * val
                hist[px,py,id2]+= fp2 * val

print(hist.shape)

histBig = np.zeros( (hist.shape[0]-1, hist.shape[1]-1, 4*8))

from numpy import linalg as LA

for i in range(hist.shape[0]-1):
    for j in range(hist.shape[1]-1):
        histBig[i,j,:9] = hist[i,j]
        histBig[i,j, 7:16] = hist[i+1,j]
        histBig[i, j, 15:24] = hist[i, j+1]
        histBig[i, j, 23:] = hist[i+1, j+1]
        histBig[i, j] = histBig[i, j]/LA.norm(histBig[i,j])


for i in range(hist.shape[0] - 1):
    for j in range(hist.shape[1] - 1):

        hist[i, j] = histBig[i, j, :9]
        hist[i + 1, j] = histBig[i, j, 7:16]
        hist[i, j + 1] = histBig[i, j, :15:24]
        hist[i + 1, j + 1] = histBig[i, j, 23:]


for px in range(int(mag.shape[0]/8)):
    for py in range(int(mag.shape[1]/8)):

        xs = px * 8 +4
        ys = py * 8 +4

        for i in range(9):

            val = hist[px,py][i]*8

            if(not val > 0):
                val = 0
            b = int( np.sqrt( val*val/(1+np.tan(i*20/180*3.14)*np.tan(i*20/180*3.14))  ) )

            a = int(np.sqrt(val*val - b*b))

            if(i<=4):
                cv2.line(img,(xs-b,ys-a),(xs+b,ys+a),(0,0,255),1)
            else:
                cv2.line(img,(xs+b,ys+a),(xs-b,ys-a),(0,0,255),1)


##hist vector
hist = histBig.flatten()

print(hist.shape)


cv2.imshow('hog visualisation',img)

cv2.imshow('magnitude hog',mag)

cv2.waitKey(0)
cv2.destroyAllWindows()