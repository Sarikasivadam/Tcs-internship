import cv2 
import os
from imutils import paths
from skimage.metrics import structural_similarity as ssim

def sift1(img1,img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1,descriptors_2)
    return len(matches)

def match(img1):
    files=list(paths.list_images('person'))
    print(files)
    # img1=cv2.imread(files[1])
    img1=cv2.resize(img1,(500,500))
    gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256]);hist1 = cv2.normalize(hist1, hist1).flatten()
    l1=[]
    results={}
    for i in files:
        img2=cv2.imread(i)
        img2=cv2.resize(img2,(500,500))
        gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # print(img2)
        # s=sift1(img1,img2)
        hist = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256]);hist = cv2.normalize(hist, hist).flatten()
        
        d = cv2.compareHist(hist1, hist, cv2.HISTCMP_CORREL)
        s=ssim(gray1,gray2)
        

        results[i] = d+s
    # sort the results
    results = sorted([(v, k) for (k, v) in results.items()], reverse = True)
    print(results)  
    k=results[0][1].split('\\')[1].split('.')[0] 
    v=results[0][0]
    print(k)
    return k,v



if __name__=="__main__":
    files=list(paths.list_images('person'))
    # files=[i.replace('\\','/') for i in files]
    print(files[1])
    img1=cv2.imread(files[2])
    img1=cv2.resize(img1,(500,500))
    gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256]);hist1 = cv2.normalize(hist1, hist1).flatten()
    l1=[]
    results={}
    for i in files:
        img2=cv2.imread(i)
        img2=cv2.resize(img2,(500,500))
        gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        print(img2)
        # s=sift1(img1,img2)
        hist = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256]);hist = cv2.normalize(hist, hist).flatten()
        
        d = cv2.compareHist(hist1, hist, cv2.HISTCMP_CORREL)
        s=ssim(gray1,gray2)

        results[i] = d+s
    # sort the results
    results = sorted([(v, k) for (k, v) in results.items()], reverse = True)
    print(results)  
    k=results[0][1].split('\\')[1].split('.')[0]  
    print(k)
    
    

