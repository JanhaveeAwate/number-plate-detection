import cv2

image= cv2.imread('C:/Users/janha/Downloads/i1.jpg')

print('Original Dimensions : ',image.shape)
 
scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)

#grayscaled image
gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image) 
cv2.waitKey(0)

#edeged image
edged = cv2.Canny(gray_image, 30, 200) 
cv2.imshow("edged image", edged)
cv2.waitKey(0)

#contour image
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=resized.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("contours",image1)
cv2.waitKey(0)

#detection
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None

i = 1
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4: 
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c) 
        new_img = resized[y:y+h, x:x+w]
        cv2.imwrite('./' + str(i) + '.png', new_img)
        i += 1
        break

if screenCnt is not None:
    cv2.drawContours(resized, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("image with detected license plate", resized)
    #cv2.waitKey(0)

    Cropped_loc = './7.png'
    cv2.imshow("cropped", cv2.imread(Cropped_loc))
    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
    print("Number plate is:", plate)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No license plate detected.")


cv2.drawContours(resized, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)
  
# Window shown waits for any key pressing event 
cv2.destroyAllWindows()
