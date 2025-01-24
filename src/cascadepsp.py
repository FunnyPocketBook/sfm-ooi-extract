import cv2
import segmentation_refinement as refine
image = cv2.imread('../data/kitchen/images/DSCF0656.JPG')
mask = cv2.imread('../output/kitchen/2025-01-23_15-45-44/mesh/mask/DSCF0656.png', cv2.IMREAD_GRAYSCALE)

# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cpu') # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False, L=900) 

# this line to save output
cv2.imwrite('output.png', output)
