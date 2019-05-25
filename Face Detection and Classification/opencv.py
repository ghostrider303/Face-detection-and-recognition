import cv2
img=cv2.imread("a.jpg")
gray=cv2.imread("a.jpg",cv2.IMREAD_GRAYSCALE)

cv2.imshow("Image:",img)
cv2.imshow("Image:",gray)
cv2.waitKey(1000)
cv2.destroyAllWindows()
