import cv2
import numpy as np

# import ipdb;ipdb.set_trace()
img=cv2.imread('./AAA_4420_.png',1)
mask=cv2.imread('./AAA_4420.png',1)
added=cv2.addWeighted(img,1,mask,0.5,0)
cv2.imwrite('added_true.jpg',added)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',added)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

for i in reversed(range(5,10)):
    print(i)

# %%
def s(nums):
    l=0
    h=len(nums)-1
    # import ipdb;ipdb.set_trace()
    while(l<=h):
        m=int((l+h)/2)
        if sum(nums[:m])==sum(nums[m+1:]):
            return m
        
        elif sum(nums[0:m])>sum(nums[m+1:]):
            h=m
        
        elif sum(nums[0:m])<sum(nums[m+1:]):
            l=m
    
    return -1

a=s([1,7,3,6,5,6])
print(a)
# %%
