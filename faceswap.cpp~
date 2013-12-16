#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

String logo_cascade_name = "haarcascade_frontalface_default.xml"; 
CascadeClassifier logo_cascade;

// detect faces
std::vector<Rect> faceDetect(Mat &frame);

// compute the depth of a common point on 2 images
float depth(int xl, int xr, float T, float f);

// copy an image onto a position in a second
void copyImage(Mat src, Mat &dst, int x, int y);

int main( int argc, const char** argv )
{
	VideoCapture cap;
	
	if(argc > 1)
	{
		cap.open(string(argv[1]));
	}
	else
	{
		cap.open(CV_CAP_ANY);
	}
	
	if(!cap.isOpened())
	{
		printf("Error: could not load a camera or video.\n");
	}
	
	// load the Haar face classifier
	if( !logo_cascade.load(logo_cascade_name) )
	{
	  printf("Error loading Haar cascade\n");
	  return -1;
	}
	
	// Matrices
	Mat face, frame, frame_gray;
	
	// create windows
	namedWindow("Face Swap", 1);
	
  // load alternative face
  face = imread("freeman.png", CV_LOAD_IMAGE_COLOR);
	cvtColor(face, face, CV_BGR2GRAY);
	
	while (1) {
	  waitKey(20);
	  cap >> frame;
	  
	  if(!frame.data)
		{
			printf("Error: no frame data from camera\n");
			break;
		}
		
		// flip for intuition
		flip(frame, frame, 1);
		
		// convert to grayscale
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    
    // detect faces
    std::vector<Rect> faces = faceDetect(frame_gray);
    
    // swap faces
    for( int i = 0; i < faces.size(); i++ )
    {
      if (faces[i].width > frame.cols/7) {
        Mat newFace = face.clone();
        resize(newFace, newFace, Size(faces[i].width, faces[i].height), 0, 0, INTER_CUBIC);
        copyImage(newFace, frame_gray, faces[i].x, faces[i].y);
      }
    }
    
    // display!
	  imshow("Face Swap", frame_gray);
	}
	
	return EXIT_SUCCESS;
}

// xl is an x-coord on the left image
// xr is an x-coord on the right image
// T is the distance between the cameras
// f is the focal length of the cameras (assumed the same)
float depth(int xl, int xr, float T, float f)
{
  return (f * T) / (float)(xl - xr);
}

std::vector<Rect> faceDetect(Mat &frame)
{ 
  std::vector<Rect> faces;
  logo_cascade.detectMultiScale( frame, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );  
  return faces;
}


void copyImage(Mat src, Mat &dst, int x, int y){
  for (int i=0; i<src.rows; i++) {
    if (y+i < dst.rows) {
      uchar *dst_row = dst.ptr(y+i);
      uchar *src_row = src.ptr(i);
      for (int j=0; j<src.cols; j++) {
        if (x+j < dst.cols) {
          if (src_row[j] > 0) {
            dst_row[x+j] = src_row[j];
          }
        }
      }
    }
  } 
}
