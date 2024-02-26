import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load your image
image = cv2.imread('/content/images.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal using GaussianBlur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding to get a binary image
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations to remove noise and improve segmentation
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area using dilatation
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that sure background is not 0 but 1
markers = markers + 1

# Mark the region of unknown with 0
markers[unknown == 255] = 0

# Apply watershed algorithm
cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red

# Display the results
plt.subplot(321), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(322), plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])

plt.subplot(323), plt.imshow(sure_bg, cmap='gray')
plt.title('Sure Background'), plt.xticks([]), plt.yticks([])

plt.subplot(324), plt.imshow(sure_fg, cmap='gray')
plt.title('Sure Foreground'), plt.xticks([]), plt.yticks([])

plt.subplot(325), plt.imshow(unknown, cmap='gray')
plt.title('Unknown Region'), plt.xticks([]), plt.yticks([])

plt.subplot(326), plt.imshow(image, cmap='gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])

plt.show()
