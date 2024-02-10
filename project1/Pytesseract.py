import cv2
import numpy as np
#import imutils

image = cv2.imread('C:/Users/janha/Downloads/i1.jpg')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

#image = imutils.resize(image, width=300)
#cv2.imshow("num.png", image)
#cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("grayed image", gray_image)
#cv2.waitKey(0)

gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
cv2.imshow("smoothened image", gray_image)
#cv2.waitKey(0)

edged = cv2.Canny(gray_image, 30, 200) 
cv2.imshow("edged image", edged)
#cv2.waitKey(0)

cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("contours",image1)
#cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None

i = 1
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4: 
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c) 
        new_img = image[y:y+h, x:x+w]
        cv2.imwrite('./' + str(i) + '.png', new_img)
        i += 1
        break

if screenCnt is not None:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("image with detected license plate", image)
    #cv2.waitKey(0)

    Cropped_loc = './7.png'
    cv2.imshow("cropped", cv2.imread(Cropped_loc))
    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
    print("Number plate is:", plate)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No license plate detected.")


cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)
#cv2.waitKey(0)

Cropped_loc = './7.png'

# Check if the cropped image file exists and is loaded successfully
cropped_image = cv2.imread(Cropped_loc)
if cropped_image is not None and cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
    cv2.imshow("cropped", cropped_image)
    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
    print("Number plate is:", plate)
else:
    print("Error: Could not load the cropped image.")

#cv2.waitKey(0)
cv2.destroyAllWindows()
