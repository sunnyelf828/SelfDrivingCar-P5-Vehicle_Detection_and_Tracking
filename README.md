# SelfDrivingCar-P5-Vehicle_Detection_and_Tracking

## Summary

In this project, I will use a set of labeled dataset for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples which come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and examples extracted from the project video itself to extract specific features for vehicles and non-vehicles, train a classifiler using Support Vector Machine and track vehicles in a video stream. The video stream comes from a camera mounted on the center of a car which recorded the real road conditions in California.



## The Project

The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
- After those first two steps, also normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. （Adding Smooth function to accelerate the calculation）
- Estimate a bounding box for vehicles detected.
- Further improvment should be work on the smooth function


[//]: #	"Image References"
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/finding_cars.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/labels_map.png

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook `Vehicle_Detection.ipynb`

I started by reading in all the labelled`vehicle` and `non-vehicle` images, a count of 8792  cars and 8968  non-cars images with size o 64x64x3 are imported. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then I use hog function in module skimage.feature to extract the Histogram of Gradient (HOG) features from the images. Here is an example of HOG extration output image with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settle down with the following

`color_space='YCrCb'`

`spatial_size=(32,32)`

`hist_bins=32`

`hist_range=(0,256)`

`orient=9`

`pix_per_cell=8`

`cell_per_block=2`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with selected HOG features, spatial features and color features.The final feature vector length is 8460. The test accuracy is  0.9885 with total test size to be 3552.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The bacis function 'find_cars()' to detect the car can be found in the file "Vehicle_Detection.ipynb". It is used to extract features using hog sub-sampling and make predictions. The hog sub-sampling helps to reduce calculation time for finding HOG features and thus provided higher throughput rate.

In the basic one, 64 was the orginal sampling rate, with 8 cells and 8 pix per cell. The step size is `cells_per_step = 2`, which means instead of overlap, we shift 2 cells each step.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. With Sliding window techniques, the target cars might be identified multiple times in  different window size. I record the positions of positive detections in each frame of the video.I used the heat map operation to take care of the multi-detection and reduce the false positive. The example images are shown below, which is basicly good.

## ![alt text][image4]![alt text][image5]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_video/project_output_video.mp4)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  



------

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The method `VehicleDetector.find_cars_smooth()`will be used to detect the car. It is basicly the same as the function `find_cars()` defined before. However, it allows the multi-scale search. More importantly, the search is optimized by processing complete frames only once every 10 frames. The restricted search is performed by appending 50 pixel to the heatmap found in last three frames. It should help to make the detection more robust and stable.

Also this model is trained with Linear SVM, more complicated classifier like CNN, SVM with nonlinear kernels can be tried