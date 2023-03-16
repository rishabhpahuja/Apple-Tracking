# Apple-Tracking
This repository is used for tracking apples in an orchard by using stereo camera setup. 

It has the following main components:

1. **Disparity:** Uses RAFT stereo to find disparity map
2. **Segmentation:** Employs UNet segmentation to segment out fruits from images
3. **Superglue:** Feature matching algorithm used in DeepSort
4. **DeepSORT:** Multi-object trackign algorithm

We started solving the problem using DeepSORT algorithm but we realised that we do not need so many components in our pipeline if we just track apples in world coordinate frame using **Extended Kalman Filter** approach. The Extended Kalman Filter effectively handles tracking during occlusion, and tracks objects in the world coordinate frame, ensuring that the coordinates of the objects being tracked remain consistent across frames allowing easy reassociation. Mahalanobis distance is used for making cost matrix to perform association using **Hungarian Algorithm**.

The diagram below shows the entire pipeline of how these components are used for tracking apples:

![Pipelien Components](flowchart.png)

The entire pipeline can be activated by runninng `wrapper.py` by running the comment 'python3 wrapper.py'. 'wrapper.py' has the following arguments:

1. **debug**: To view and save images of each component
2. **disparity_model**: To give path of weights for RAFT stereo
3. **left**: To give path for left stereo image
4. **right**: To give path for right stereo image
5. **size**: To give downsampling image size to run RAFT stereo faster
6. **seg_mode**: To give path for weights of UNet segmentation
7. **scale**: To give downsamplign factor for segmentation
8. **match**: Bool value whether to perform histogram equilization of the inputted images with a reference (Used in cases when the lighting onditions of images used to train segmentation model is very different from the inputted images)

## Results:

{% figure caption:"Realtime Apple Tracking" %}
    ![Realtime Apple Tracking](apple_tracker.gif)
{% endfigure %}
![Realtime Implementation](apple_tracker.gif) ![Point Cloud Representation](pc.gif)





 
