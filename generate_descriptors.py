import cv2
import numpy as np
import os

def create_descriptors(dir):
    images_paths=[]
    for dirname,dirnames,filename in os.walk(dir):
        images_paths.extend(filename)

    sift = cv2.xfeatures2d.SIFT_create()

    for image_path in images_paths:
        path=os.path.join(dir,image_path)
        img=cv2.imread(path,0)
        kp, d = sift.detectAndCompute(img,None)
        d_file=image_path.replace('PNG','npy')
        np.save(os.path.join(dir,d_file),d)


if __name__=="__main__":
    create_descriptors('tattoos')