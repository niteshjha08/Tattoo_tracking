import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def drawViableMatches(img,kp,image_name,tattoos_dir,sift,viable_match):
    image=cv2.imread(os.path.join(tattoos_dir,image_name),0)
    keypoints,descriptor= sift.detectAndCompute(image,None)
    img3=cv2.drawMatchesKnn(img,kp,image,keypoints,viable_match,img)
    plt.imshow(img3), plt.show()


def scan(suspects_dir,tattoos_dir):
    suspects=[]
    for dir,subdir,files in os.walk(suspects_dir):
        suspects.extend(files)
    print(suspects)
    idx=input("Enter index of suspect to scan database [0-{}]:".format(len(suspects)-1))
    suspect=suspects[int(idx)]
    path=os.path.join(suspects_dir,suspect)
    img=cv2.imread(path)

    sift=cv2.xfeatures2d.SIFT_create()
    kp, d = sift.detectAndCompute(img,None)

    FLANN_INDEX_KDTREE=0
    indexParam=dict(algorithm= FLANN_INDEX_KDTREE, trees= 5)
    searchParam=dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParam,searchParam)
    MIN_MATCH_COUNT=10


    files=[]
    descriptors=[]
    image_list=[]
    for dir,subdir,filename in os.walk(tattoos_dir):
        files.extend(filename)
        for f in files:
            if(f.endswith('npy')):
                descriptors.append(f)
            if(f.endswith('PNG')):
                image_list.append(f)

    max_matches=0
    k=0
    count=0
    for i in descriptors:
        viable_match=[]
        matches=flann.knnMatch(d,np.load(os.path.join(tattoos_dir,i)),k=2)
        for m,n in matches:
            if(m.distance<0.7*n.distance):
                viable_match.append(matches)
        if(len(viable_match))>MIN_MATCH_COUNT:
            print('{0} is a match! It had {1} viable_matches'.format(i,len(viable_match)))
        else:
            print('{0} is not a match! It had {1} viable_matches'.format(i,len(viable_match)))
        if(len(viable_match)>max_matches):
            k=i
            max_matches=len(viable_match)
    # Optional: For visualization
        drawViableMatches(img,kp,image_list[count],tattoos_dir,sift,matches)
    print("The culprit is : {}".format(k).upper())

if __name__=="__main__":
    suspects_dir='suspects'
    tattoos_dir='tattoos'
    scan(suspects_dir,tattoos_dir)