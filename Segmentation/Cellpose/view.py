import matplotlib.pyplot as plt
import cv2

image = cv2.imread('cellular-main/experiments/combined_masks-3/Timepoint_004_220518-ST_K07_s3_masks.png', -1)  # -1 for unchanged
plt.imshow(image, cmap='gray')  # Choose colormap as needed
plt.colorbar()  # Optional: to see the value range on a colorbar
plt.show()


