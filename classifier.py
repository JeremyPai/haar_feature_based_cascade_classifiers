import cv2

# Load the image for obeject detection (in this case, human is what we want to detect)
image = cv2.imread("image.jpg")

# Need to convert to grayscale for inputting to Harr Cascade (Must)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade Detector (pre-trained)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Set the parameter of detector and return the detection results
rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(15,15))

# Loop over every bounding box and draw a rectangle around all people
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 5)

# Show final result
cv2.imshow("People found!", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite("Detection.jpg", image)