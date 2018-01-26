## Writeup 
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[imageHogPara]: ./output_images/HOG_Para.png
[image_s1]: ./output_images/sacle_1.png
[image_s2_5]: ./output_images/sacle_2_5.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/heatmap.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (Cell 1 in `project.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, below are the results. I picked YCrCb, All Channels for faster training time, and fast predict time. (Cell 5 in `project.ipynb` with 1000 samples)

![alt text][imageHogPara]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using colorspace YCrCb, 9 orientations 8 pixels per cell and 2 cells per block (Cell 5 in `project.ipynb`)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at the upper part of the image using smaller scall for car further away, using larger scale for closer car which is on the lower part of the image) (Cell 7-10 in `project.ipynb` )

E.g.
ystart = 400, ystop = 480, scale = 1
ystart = 400, ystop = 500, scale = 1.5
ystart = 400, ystop = 550, scale = 2
ystart = 400, ystop = 600, scale = 2.5

Scale 1
![alt text][image_s1]

Scale 2.5
![alt text][image_s2_5]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

Heatmap

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I am using 90 frames average (3 secs) to for false positives.(Cell 16 in `project.ipynb` )

heat_history = reduce(lambda h, acc: h + acc, hist.rect)/90

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Current preformance is slow. To increase the preformance, we can search less windows in X-axis for different part of the image
* We are using car at the moment, there will be other vehicle such as bigger trucks and bike. Which it cannot detect


