import cv2
import matplotlib.pyplot as plt
import numpy as np

def not_rotated(img_gray,template):
  
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.96
    loc = np.where( res >= threshold)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        print(pt,(pt[0] + w, pt[1] + h))
        
        #Four points
        print("----------------------------")
          
        print("Four Points:")
        print("statr ppoint", pt)
        print("up_right: ", pt[0] + w, pt[1])
        print("bot_left : ", pt[0], pt[1] + h)
        print("but_rigt:", pt[0] + w, pt[1] + h)
        
        
    cv2.imshow('Detected',img_rgb)
    cv2.imwrite("detected.jpg",img_rgb)

def with_rotated(img1,img2):
    
    MIN_MATCH_COUNT = 2
    
    ## Create ORB object and BF object
    orb = cv2.ORB_create()
    
    
    # Find the keypoints and descriptors 
    kpt1, des1 = orb.detectAndCompute(img1,None)
    kpt2, des2 = orb.detectAndCompute(img2,None)
    
    # match descriptors and sort them 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    dmatches = sorted(matches, key = lambda x:x.distance)
    
    ## extract the matched keypoints
    src_pts  = np.float32([kpt1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpt2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
    
    ## find homography matrix and do perspective transform
    #using findHomography ve perspectiveTransform for rotated  image 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    ## draw found regions
    #afer perspectiveTransform give again orjinal image
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,255,255), 1, cv2.LINE_AA)
    #gray3 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Detected_image", img2)
    cv2.imwrite("Detected_image.jpg",img2)
    
    ## draw match lines
    res = cv2.drawMatches(img1, kpt1, img2, kpt2, dmatches[:20],None,flags=2)
    
    cv2.imshow("Matches", res);
    
    cv2.waitKey();cv2.destroyAllWindows()
    
      
    #Four points
    print(img2.shape)
    print("-----------------------")
    w = img2.shape[1]
    h = img2.shape[0]
    
    print("Four Points:")
    print("Start Point : ", [0,0]  ) 
    print("up_right :" ,[img2.shape[1],0])
    print("bot_let", [0,img2.shape[0]])
    print("bot_right", [img2.shape[1],img2.shape[0]])

# main
if __name__ == "__main__":
    
    #not_rotated
    template = cv2.imread('Small_area.png',0)
    w, h = template.shape[::-1]
    img_rgb = cv2.imread('StarMap.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print("not_rotated")
    not_rotated(img_gray,template)
    
    print("---------------------------------------")
    
    #with_rotated
    img1 = cv2.imread("Small_area_rotated.png",0)
    #img2 = cv2.imread("StarMap.png",0)
    print("With_Rotated")
    with_rotated(img1,img_gray)
    
else:
    pass