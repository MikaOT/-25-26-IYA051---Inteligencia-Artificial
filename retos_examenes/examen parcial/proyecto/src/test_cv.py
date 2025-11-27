import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

img = np.zeros((300, 300, 3), np.uint8)
cv2.putText(img, "OK", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
