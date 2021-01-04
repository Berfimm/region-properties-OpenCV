import cv2
import numpy as np

img = cv2.imread('Resources/son1.png')

#input and output image
def getContours(img,imgContour):
    #Find contours and set contour retrivial mode
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    #cnt is the one of the contours for every object in image
    for cnt in contours:

        # Contour for shape and I find area and length using this contour
        cv2.drawContours(imgContour, cnt,-1, (0, 255, 0), 1)

        (x_cen,y_cen),radius =cv2.minEnclosingCircle(cnt)
        center = np.array([[int(x_cen),int(y_cen)]])
        radius = int(radius)

        #BoundingBox(Rectangle):gives the bounding box parameters => (x,y,width,height)
        # box = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(box)
        # box = np.array(box, dtype='int')
        # cv2.drawContours(imgContour, [box], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(imgContour,(x, y),(x+w, y+h),(0,255,0),2)
        print("bounding_box(x,y,w,h):", x, y, w, h)

        #The extreme points
        l_m = tuple(cnt[cnt[:, :, 0].argmin()][0])
        r_m = tuple(cnt[cnt[:, :, 0].argmax()][0])
        t_m = tuple(cnt[cnt[:, :, 1].argmin()][0])
        b_m = tuple(cnt[cnt[:, :, 1].argmax()][0])
        pst = [l_m, r_m, t_m, b_m]

        # W: and H: texts (Length)
        cv2.putText(imgContour, "w={},h={}".format(w, h), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    16)

        # give us specific contour (cnt) area
        area = cv2.contourArea(cnt)


        # Area text
        cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, 16)



        #Find major and minor length
        argvalues= np.array(pst)
        center_array = np.full((argvalues.shape[0],argvalues.shape[1]), center, dtype = int)
        ma= argvalues - center_array
        ma2 = np.sqrt(np.square(ma[:, 0]) + np.square(ma[:,1]))
        major =int(np.amax(ma2))
        minor = int(np.amin(ma2))

        #MajorAxislength,MinorAxisLength,Eccentricity
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img,ellipse,(0, 0, 255), 2)

        #semi-major and semi-minor
        a = major / 2
        b = minor / 2

        #Formula of eccentricity is :
        Eccentricity = round(np.sqrt(pow(a, 2) - pow(b, 2))/a, 2)

        x = int(x + w / 2) + 1
        y = int(y + h / 2) + 1

        # print(minorAxisLength)
        # print(majorAxisLength)
        # print(Eccentricity)

        cv2.putText(imgContour, 'Minor ='+str(round(minor, 2)), (x+10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,16)
        cv2.putText(imgContour, 'Major ='+str(round(major, 2)), (x+10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 16)
        cv2.putText(imgContour, 'Eccentricity ='+str(round(Eccentricity, 3)), (x+10, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 16)


imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (7, 7),1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
ImgThresh = cv2.threshold(imgGray, 50 , 50, cv2.THRESH_BINARY)[1]
imgCanny = cv2.Canny(imgGray,50,50)
getContours(imgCanny,imgContour)

cv2.imshow("Original Image", img)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Contour", imgContour)
cv2.waitKey(0)
cv2.destroyAllWindows()