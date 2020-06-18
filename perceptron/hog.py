import numpy as np
import cv2

hog = cv2.HOGDescriptor('./hog.xml')

def get_hog_feat(im):
    return hog.compute(im)

if __name__ == '__main__':
    im = np.random.randint(0, 255, size=(28,28), dtype=np.uint8)
    print (get_hog_feat(im).shape)
