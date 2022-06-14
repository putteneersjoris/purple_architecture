import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
# Load image and keep a copy
# image = cv2.imread("C:/Users/joris/OneDrive/Documenten/creatievemakers/summerschool/prepwork/opencv_houdini/download.png"); orig_image = image.copy()

# plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
# plt.title('House')

# # plt.show()

# # Grayscale and binarize
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# # Find contours 
# contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


# Iterate through each contour and compute the bounding rectangle
# for c in contours:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    
    
#     plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
#     # plt.title('Bounding Rectangle'); plt.show()

# # Iterate through each contour and compute the approx contour
# for c in contours:
#     # Calculate accuracy as a percent of the contour perimeter
#     accuracy = 0.03 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, accuracy, True)
#     cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Approx Poly DP'); plt.show()
'''


# Load image and keep a copy
# image = cv2.imread("C:/Users/joris/OneDrive/Documenten/creatievemakers/summerschool/prepwork/opencv_houdini/a.jpg"); 
# orig_image = image.copy()


path = "./photoshop_files/a_"

cap = cv2.VideoCapture(path+"%01d.jpg", cv2.CAP_IMAGES)

test = 0
while True:
    ret, frame = cap.read()
    test +=1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 255, 255, 255)

    # Find contours 
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
    # Sort Contors by area and then remove the largest frame contour
    n = len(contours) - 1
    contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]


    # Iterate through contours and draw the convex hull
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(frame, [hull],-2, (0, 0, 255), thickness=-1)
        # cv2.fillPoly(frame, [hull], color=(255,0,0))
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow("roi",cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('./opencv_files/image_convex_hull_' +str(test)+'.png',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
