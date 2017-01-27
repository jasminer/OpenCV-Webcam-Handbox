#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


Ptr < BackgroundSubtractor > pMOG2;
class roiContour{
public:
	
	vector<cv::Point>  contour;
	bool upper;
	int minX;
	int minY;
	int maxX;
	int maxY;


	roiContour(vector<cv::Point> ctr) :contour(ctr), upper(false){}
	void compute(int matrixWidth, int matrixHeight){
		if (contour.empty())
			return;
		
		minX = maxX = contour[0].x;
		minY = maxY = contour[0].y;
		for (int i = 1; i < contour.size(); i++){
			if (contour[i].x < minX)
				minX = contour[i].x;
			if (contour[i].y < minY)
				minY = contour[i].y;
			if (contour[i].x > maxX)
				maxX = contour[i].x;
			if (contour[i].y > maxY)
				maxY = contour[i].y;

			
		}
		if (maxY < matrixHeight / 2)
			upper = true;

	}
};
int main()
{
	//waitKey(0);                                          // Wait for a keystroke in the window
	//return 0;
	
	Mat frame, resizeF, gray, fgMaskMOG2, edges;
	VideoCapture cam("vid.mp4");
	cam.set(CV_CAP_PROP_POS_FRAMES, 130);
	//VideoCapture cam(0);
	namedWindow("Image");
	namedWindow("MOG2");
	pMOG2 = new BackgroundSubtractorMOG();
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	//Mat elementBig = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	int erosion_size = 1;
	Mat elementErode = getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	int dilation_size = 15;
	Mat elementDilate = getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		cv::Point(dilation_size, dilation_size));
	Mat hsv;
	Mat hsandv[3];
	Mat hsvBinarized;
	cam.read(frame);
	while (cam.read(frame)&&cam.isOpened())
	{
		if (frame.empty()){
			continue;
		}
		//flip(frame, frame, 1);
		resize(frame, frame, Size(frame.size().width / 3, frame.size().height / 3));
		//resize(frame, frame, Size(frame.size().width / 2, frame.size().height / 2));
		cvtColor(frame, gray, CV_BGR2GRAY);
		cvtColor(frame, hsv, CV_BGR2HSV);

		split(hsv, hsandv);

		inRange(hsandv[0], 19, 240, hsvBinarized);
		bitwise_not(hsvBinarized, hsvBinarized);
		pMOG2->operator()(gray, fgMaskMOG2);

		
		vector<vector<cv::Point> > contours1;
		vector<Vec4i> hierarchy1;
		vector<Vec4i> lines;

		
		int scale = 1;
		int delta = 4;
		int ddepth = CV_16S;

		Mat grad, grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		
		/// Gradient X
		//Sobel(edges, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_CONSTANT);
		//convertScaleAbs(grad_x, abs_grad_x);
		/// Gradient Y
		//Sobel(edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_CONSTANT);
		//convertScaleAbs(grad_y, abs_grad_y);

		/// Total Gradient (approximate)
		//addWeighted(abs_grad_x, 0, abs_grad_y, 1, 0, grad);

		
		/*
		Mat cannyGrad, canniedGrad, houghGrad, binarizedLines;

		cvtColor(frame, binarizedLines, COLOR_BGR2GRAY); // to make sizes same
		binarizedLines = Scalar::all(0); // to make it black

		// go ahead and play around with these factors... or you can adjust them on runtime
		int iHoughLinesThreshold = 280, iHoughLinesMinLineSize = 280,
			iHoughLinesGap = 200;


		try{
			//cvtColor(grad, cannyGrad, CV_GRAY2BGR);
			
			GaussianBlur(fgMaskMOG2, cannyGrad, Size(5, 5), 2, 2);

			Canny(cannyGrad, canniedGrad, 45, 75);
			//string type = type2str(canniedGrad.type());
			//cout <<endl<< type<< endl;
			HoughLinesP(canniedGrad, lines, 1, CV_PI / 180, iHoughLinesThreshold, iHoughLinesMinLineSize, iHoughLinesGap);
		}

		catch (cv::Exception ex){
			cout << ex.msg;
		}
		for (int i = 0; i < lines.size(); ++i) {
			cv::Vec4i coordinate = lines[i];

			const cv::Point pt1(coordinate[0], coordinate[1]);
			const cv::Point pt2(coordinate[2], coordinate[3]);

			const cv::Scalar color(0, 0, 255);
			const int thickness = 3;
			const int lineKind = 8;
			const int shift = 0;
			//cv::line(img, pt1, pt2, color, thickness, lineKind, shift);
			line(binarizedLines, pt1, pt2, Scalar(255), 1, lineKind, shift);
			line(frame, pt1, pt2, Scalar(0, 0, 255), 5, lineKind, shift);
		}
		*/
		
		
		bitwise_and(fgMaskMOG2, hsvBinarized, fgMaskMOG2);
		imshow("MOG2", fgMaskMOG2);
		//morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_OPEN, element);
		erode(fgMaskMOG2, fgMaskMOG2, elementErode);
		
		imshow("MOG3", fgMaskMOG2);
		Canny(fgMaskMOG2, edges, 5, 30);
		//morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_CLOSE, elementBig);
		dilate(fgMaskMOG2, fgMaskMOG2, elementDilate);

		//erode(fgMaskMOG2, fgMaskMOG2, elementErode); 
		
		imshow("MOG4", fgMaskMOG2);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(fgMaskMOG2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		/// Draw contours
		//Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
		cout << contours.size() << " contours found: ";
		int cSize = contours.size();
		vector<roiContour> roiContours;
		
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
		cout << endl;
		Mat roi;
		bool upper=false, lower=false;
		int upperI, lowerI;
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
					cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
					//imshow("Roi", roi);
				}
			}
		}
		catch (cv::Exception ex){
			cout << ex.msg<<endl;
		}
		

		// one contour greater than 400 above center and same below

		/*
		
		cv::Mat image ;
		cv::Mat roi = image(cv::Rect(100, 100, 300, 300));
		cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 125, 125));
		double alpha = 0.3;
		cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi);
		
		*/


		// color code: #A67FC4
		imshow("Edges", edges);
		imshow("Skin", hsvBinarized);
		imshow("Image", frame);
		
		
		if (waitKey(5) > 0){
			break;
		}
	}

	return 0;

}
