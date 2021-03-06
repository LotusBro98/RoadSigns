#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <string>
#include <dirent.h>

void Sobel3(cv::Mat& src, cv::Mat& dst)
{
	double koeff = 0.2;

	std::vector<cv::Mat> planes;
	cv::split(src, planes);

	dst = cv::Mat::zeros(src.size(), planes[0].type());

	for (int i = 0; i < 3; i++)
	{
		cv::Mat grad_x;
		cv::Mat grad_y;

		cv::Sobel(planes[i], grad_x, CV_16S, 1, 0);
		cv::Sobel(planes[i], grad_y, CV_16S, 0, 1);

		cv::convertScaleAbs(grad_x, grad_x);
		cv::convertScaleAbs(grad_y, grad_y);

		cv::addWeighted(dst, 1, grad_x, koeff, 0, dst);
		cv::addWeighted(dst, 1, grad_y, koeff, 0, dst);
	}
}

int blurSize = 3;
int minSat = 180;
double minContourArea = 200;
double contourApproxEpsInPeri = 0.1;
double minCircRatio = 0.9;

int croppedSize = 256;

enum
{
	CONTOUR_CIRCLE,
	CONTOUR_TRIANGLE,
	CONTOUR_QUADRANGLE,
	CONTOUR_OTHER
};

int processContour(std::vector<cv::Point>& contour, cv::Point& center,  double& radius)
{
    convexHull(contour, contour);
	
	double area = cv::contourArea(contour);
	double peri = cv::arcLength(contour, true);

	cv::Moments M = cv::moments(contour);
	center = cv::Point(M.m10 / M.m00, M.m01 / M.m00);
	radius = sqrt(area / M_PI);

	if (area < minContourArea)
		return CONTOUR_OTHER;

	double circRatio = area * 4 * M_PI / peri / peri;

	if (circRatio > minCircRatio)
		return CONTOUR_CIRCLE;

	cv::approxPolyDP(contour, contour, peri * contourApproxEpsInPeri, true);

	if (contour.size() == 3)
		return CONTOUR_TRIANGLE;
	else if (contour.size() == 4)
		return CONTOUR_QUADRANGLE;

	return CONTOUR_OTHER;
}

void sliceCircle(cv::Mat& image, cv::Point center, double radius, cv::Mat& slice)
{
	cv::Mat M = cv::getAffineTransform(
			std::vector<cv::Point2f>{
				center + cv::Point(-radius, -radius),
				center + cv::Point(radius, -radius),
				center + cv::Point(radius, radius),
				},
			std::vector<cv::Point2f>{
				{0, 0},
				{(float)croppedSize, 0},
				{(float)croppedSize, (float)croppedSize},
			}
	);

	cv::warpAffine(image, slice, M, cv::Size(croppedSize, croppedSize));
}

int getMaxXY(std::vector<cv::Point>& contour)
{
	int maxI = 0;
	double maxXY = 0;
	for (int i = 0; i < contour.size(); i++)
	{
		double XY = contour[i].x + contour[i].y;
		if (XY > maxXY)
		{
			maxI = i;
			maxXY = XY;
		}
	}
	return maxI;
}

void sliceTriangle(cv::Mat& image, std::vector<cv::Point>& contour, cv::Mat& slice)
{
	int max = getMaxXY(contour);

	cv::Mat M = cv::getAffineTransform(
			std::vector<cv::Point2f>{
				contour[(0+max) % 3],
				contour[(1+max) % 3],
				contour[(2+max) % 3]
			},
			std::vector<cv::Point2f>{
				{(float)croppedSize, (float)croppedSize},
				{0, (float)croppedSize},
				{(float)croppedSize / 2, 0},
			}
	);

	cv::warpAffine(image, slice, M, cv::Size(croppedSize, croppedSize));
}

void sliceQuadrangle(cv::Mat& image, std::vector<cv::Point>& contour, cv::Mat& slice)
{
	int max = getMaxXY(contour);

	cv::Mat M = cv::getPerspectiveTransform(
			std::vector<cv::Point2f>{
				contour[(0+max) % 4],
				contour[(1+max) % 4],
				contour[(2+max) % 4],
				contour[(3+max) % 4]
			},
			std::vector<cv::Point2f>{
				{(float)croppedSize, (float)croppedSize},
				{0, (float)croppedSize},
				{0, 0},
				{(float)croppedSize, 0},
			}
	);

	cv::warpPerspective(image, slice, M, cv::Size(croppedSize, croppedSize));

}

int main()
{
	int sliceI = 0;

	DIR* dir = opendir("../images");
	struct dirent *ent;

	while ((ent = readdir(dir)) != NULL)
	{
		std::string path = std::string("../images/") + ent->d_name;
		if (strcmp(ent->d_name, "..") == 0 || strcmp(ent->d_name, ".") == 0)
			continue;

		std::cout << path << "\n";

		cv::Mat image = cv::imread(path);
		cv::Mat imageDisp;
		image.copyTo(imageDisp);

		cv::Mat hsv;
		cv::Mat sat;
		cv::GaussianBlur(image, image, cv::Size(blurSize, blurSize), 1);
		cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
		cv::extractChannel(hsv, sat, 1);
		
		cv::inRange(sat, minSat, 255, sat);
		cv::imshow("Saturation", sat);

		std::vector<std::vector<cv::Point> > contours;
		findContours(sat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
		std::vector<std::vector<cv::Point> >hull( contours.size() );
		for( int i = 0; i < contours.size(); i++ )
		{
			std::vector<cv::Point> contour = contours[i];
			cv::Point center;
			double radius;

			cv::Mat slice;
			
			int type = processContour(contour, center, radius);
			if (type == CONTOUR_OTHER)
				continue;
			else if (type == CONTOUR_CIRCLE) {
				cv::circle(imageDisp, center, radius, cv::Scalar(0, 100, 255), 3);
				sliceCircle(image, center, radius, slice);

			}
			else if (type == CONTOUR_TRIANGLE)
			{
				cv::polylines(imageDisp, contour, true, cv::Scalar(0, 255, 0) , 3);
				sliceTriangle(image, contour, slice);
			}
			else if (type == CONTOUR_QUADRANGLE)
			{
				cv::polylines(imageDisp, contour, true, cv::Scalar(255, 100, 0) , 3);
				sliceQuadrangle(image, contour, slice);
			}

			//cv::imshow("Slice " + std::to_string(sliceI), slice);
			sliceI++;

			cv::imwrite("../dataset/unsorted/" + std::to_string(sliceI) + ".jpg", slice);
		}

		cv::imshow("Image", imageDisp);
		cv::waitKey();

	}

	return 0;
}
