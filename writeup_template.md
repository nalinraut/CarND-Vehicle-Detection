---
[//]: # (Image References)
[image0]: ./assets/video.gif
[image1]: ./assets/vehicles.png
[image2]: ./assets/Original_YCrCb_HOG.png
[image3]: ./assets/hogfeatures.png

[image4]: ./assets/ibw.png
[image5]: ./assets/test1.png
[image6]: ./assets/test2.png
[image7]: ./assets/test3.png
[image8]: ./assets/test4.png
[image9]: ./assets/test5.png
[image10]: ./assets/test6.png
[image10]: ./assets/win.png




![alt text][image0]

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.





[video1]: ./project_video.mp4
---


### Histogram of Oriented Gradients (HOG)

#### 1.  HOG features.

  
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of `vehicle` and `non-vehicle` classes:

![alt text][image1]

Below is the piece of code for extracting HOG features from an image. I extract HOG features from each channel of the image as shown below and concatenate them.

```python
    def getHOGFeatures(self,image):
        """
            HOG of every channel of given image is computed and
            is concatenated to form one single feature vector
        """
        HOG_ch1 = hog(image[:,:,0], 
                            orientations= self.orientations , 
                            pixels_per_cell= self.pixels_per_cell , 
                            cells_per_block= self.cells_per_block,
                            visualise=False)
        HOG_ch2 = hog(image[:,:,1], 
                            orientations= self.orientations , 
                            pixels_per_cell= self.pixels_per_cell , 
                            cells_per_block= self.cells_per_block,
                            visualise=False)
        HOG_ch3 = hog(image[:,:,2], 
                            orientations= self.orientations , 
                            pixels_per_cell= self.pixels_per_cell , 
                            cells_per_block= self.cells_per_block,
                            visualise=False)
        return np.concatenate((HOG_ch1, HOG_ch2, HOG_ch3))

```

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image2]



#### 2. HOG parameters.

The scikit-image hog() function takes in a single color channel or grayscaled image as input, as well as various parameters. These parameters include `orientations`, `pixels_per_cell` and `cells_per_block`.

The number of `orientations` is specified as an integer, and represents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins. I set the value of orientatons to 8 since it yields a better accuracy.

The `pixels_per_cell` parameter specifies the cell size over which each gradient histogram is computed. This paramater is passed as a 2-tuple so you could have different cell sizes in x and y, but cells are commonly chosen to be square.I choose a square of 8 pixel by 8 pixel.

The `cells_per_block` parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given cell will be normalized. Block normalization is not necessarily required, but generally leads to a more robust feature set. For this, I chose a value of (2,2)


#### 3. Classifier.

Random forest classifier performs better than the SVM classifier considering time and accuracy.Therefore, I used `RandomForestClassifier` from `sklearn.ensemble`. The foolowing Feature vector was used to train the classifier.
Following are raw and normalised HOG features for the corresponding image shown. The feature vector is used for training the classifer. 

![alt text][image3]

### Sliding Window Search

#### Implementation

I tried to scale the window in four sizes, to detect car near as well as far way. The four scales are shown below 
```python
def get_windows(self,image):
        """
            pre-defined window sizes
        """
        window_image = np.copy(image)
        height, width,_ = window_image.shape

        # print(width,height)
        scale_factors = [
                        (0.4,1.0,0.55,0.8,64),
                        (0.4,1.0,0.55,0.8,96),
                        (0.4,1.0,0.55,0.9,128),
                        (0.4,1.0,0.55,0.9,140),
                        (0.4,1.0,0.55,0.9,160),
                        (0.4,1.0,0.50,0.9,192)]

        windows = list()
        for scale_factor in scale_factors:
            window_1 = self.slide_window(window_image,
                            x_span=[int(scale_factor[0]*width), 
                                            int(scale_factor[1]*width)], 
                            y_span=[int(scale_factor[2]*height), 
                                            int(scale_factor[3]*height)],
                            
                            xy_window=( scale_factor[4], 
                                        scale_factor[4]), 
                            xy_overlap=(0.5, 0.5))
            windows.append(window_1)
        

        return windows
```
The window overlap was chosen to be 50%.

![alt text][image11]

#### 2. Example Images

I used the above mentioned parameter and window sizes to obtain features. I then created a heatmap of the bounding boxes predicted by the classifier, thresholded it and then created blobs to find the final bounding boxes. 

![alt text][image4]

---

### Video Implementation

#### 1. Video Link
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.

##### Here are six frameswith windows, their corresponding heatmaps, and final images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]





---

### Discussion

The pipeline is too slow as compared to deep learning methods. The pipeline would fail in realtime situations. Deep learning methods can be used for vehicle detection. Methods such as Faster RCNN, Yolo, SSDs will help in 2D object recognition. For better results 3DCNN, MV3D, and other fusion networks can be used to fuse 3D lidar data and image data for vehicle detection and tracking.


