# OpenCV-Webcam-Handbox

1	Introduction
This project is a Computer vision project that detects human hands from a still live or recorded video. The project may be used in various scenarios where human hand detection and augmenting objects in motion is relevant.

2	Step 1: Detecting Color
The proposed algorithm follows a two-level detection technique that improves its accuracy. These processes are repeated and reiterated on each camera / video frame.

2.1 	Color Space
Initially, skin detection is performed based on a famous paper on skin detection using HSV color space.

2.1.1  	HSV Color space:
The object is transformed to HSV color space to filter out extra parts of image except hand skin.
HSV (Hue, Saturation and Value) – defines a type of color space. It is similar to the modern RGB and CMYK models. The HSV color space has three components: hue, saturation and value. ‘Value’ is sometimes substituted with ‘brightness’ and then it is known as HSB. The HSV model was created by Alvy Ray Smith in 1978. HSV is also known as the hex-cone color model. To achieve better accuracy because filtering out a color from HSV is more efficient.

2.1.2 	Skin Color Range:
We defined a color range based on relevant research in skin detection in the HSV color space—Oliveira, Skin Detection using HSV color space (2009).[1]
The following code converts an RGB image in to HSV color space:
cvtColor(frame, hsv, CV_BGR2HSV);
split(hsv, hsandv);
The ‘hue’ component of hsv is taken and applied color based range filtering using OpenCV’s function inRange() according to the range determined by the Oliveira paper.
inRange(hsandv[0], 19, 240, hsvBinarized);
bitwise_not(hsvBinarized, hsvBinarized);

2	 Step 2: Removing Elements in Real-Time
After the completion the first level, we implement motion detection using Background Subtraction algorithm.
2.1	Background Subtraction:
A common approach to detect any type of motion in a still video is to perform background subtraction, which identifies moving objects from the portion of a video frame that differs significantly from a background model. There are many challenges in developing a good background subtraction algorithm. First, it must be robust against changes in illumination. Second, it should avoid detecting non-stationary background objects such as distractions in camera pixels.
After deciding to use background subtraction we selected from two ready-made functions in OpenCV--- MOG and MOG2—which are two forms of background subtraction. In this case, MOG2 is more suited because it renders a more refined color.
pMOG2->operator()(gray, fgMaskMOG2);
The above snippet is responsible for running BGSubtraction.
Next, a grayscale frame named “gray” is fed to the operator() function which in turn stores the detection result in fgMaskMOG2.

2.2	Combining Results
The following line of code is applied bitwise and to the results of both the Skin detection and Background Subtraction.
bitwise_and(fgMaskMOG2, hsvBinarized, fgMaskMOG2);
hsvBinarized stores the pixels where skin is detected, and fgMaskMOG2 stores the pixels where motion is detected, so after taking their and product, now fgMaskMOG2 contains pixels where both skin and motion are detected.
2.2.1  	Image Refinement
erode(fgMaskMOG2, fgMaskMOG2, elementErode);
dilate(fgMaskMOG2, fgMaskMOG2, elementDilate);
The lines above do erosion and dilation which are technique of image refinement, they decrease extra dots or stray pixels detected using background subtraction and skin detection to refine the output of Background Subtraction and HSV skin detection.

2.2.2  	Canny Edge Detection
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. The following line is running canny edge detection algorithm:
Canny(fgMaskMOG2, edges, 5, 30);
It basically detects all the opaque objects in the image/frame and detects its edges. (Minimum and maximum distance between two adjacent edges be 5 and 30) 
2.2.3  	Gaussian Blurring
Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function. It is a widely used effect in graphics software, typically to reduce image noise.
2.2.4  	Contouring
We have detected edges of all distinct skin objects that are moving or that appeared into the frame from out of the frame. Now we use contouring to find contours of all the objects. Contour is just an array of points in the form of (x,y) ordered pair. Each contour is an array of Points. Each point has two components x and y like in a real world 2d space.
So the following function is used to find contours:
findContours(fgMaskMOG2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
So all the contours (if found) are stored in contours named object which is a vector of contours (each contour is a vector of points )

3	Augmentation Box Between Hands
Up to this point we have got all the contours of skins detected using two level algorithms explained above.
Now we have to determine whether or not the detected contours are hands of a person, following code does so:

for (int i = 0; i< cSize; i++)
		{
//drawContours(frame, contours, i, Scalar(255,255,255), 2, 8, hierarchy, 0, Point());
				float area = contourArea(contours[i]);
				cout<< area <<", " ;
					if (area > 400){
						roiContour ctr(contours[i]);
						ctr.compute(frame.cols, frame.rows);
						roiContours.push_back(ctr);
								}
					if (i == cSize - 1){ // last contour
				
								}
		}
    
The code above loops through all the detected contours and then if the area of contours is greater than 400 (a standard set manually that a person’s detectable hand must have a minimum area of 400, you may pay around with this depending on changing situations) then computes whether or not the contour is above or below from the mid point of the frame.

try{
		if (roiContours.size() >= 2){
for (int i = 0; i < roiContours.size(); i++){
				upper = roiContours[i].upper;
					if (upper) upperI = i;
					else { lower = true; lowerI = i; }
							}
					if (upper && lower){
roi = frame(cv::Rect(Point(roiContours[upperI].minX, roiContours[upperI].maxY), Point(roiContours[lowerI].maxX, roiContours[lowerI].minY)));
Mat color(roi.size(), CV_8UC3, cv::Scalar(167, 127, 197));
						double alpha = 0.6;
cv::addWeighted(color, alpha, roi, 1.0 -alpha, 0.0, roi);
						//imshow("Roi", roi);
								}
							}
						}
This code snippet above determines if both uthe pper side hand is detected and lower side hand is detected then subsequently draws a rectangle between both the hands (contours) using addWeighted function.
cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);


