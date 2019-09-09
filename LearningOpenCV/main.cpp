
#include <iostream>
#include <exception>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <thread>

using namespace cv;
using namespace std;

void menu(int &option) {

	cout << "\n\n\n\t\t\t\t*****************************" << endl
		<< " " << endl
		<< "\t\t\t\t Opcion 1: Abrir una imagen " << endl
		<< "\t\t\t\t Opcion 2: Abrir camera " << endl
		<< "\t\t\t\t Opcion 3: Ambos " <<endl
		<< "\t\t\t\t Opcion 4: Face Detection " << endl
		<< "\t\t\t\t Opcion 0: Salir " << endl
		<< " " << endl
		<< "\t\t\t\t*****************************" << endl
		<< " " << endl
		<< " Escribasu opción: " ;

	cin >> option;

}

void ReadImage(int argc, char** argv) {
	cout << "Abriendo imagen lenna.png" << endl;
	
	Exception ex;

	String imageName("C:/Users/aleja/Pictures/image.jpg"); // by default
	if (argc > 1)
	{
		imageName = argv[1];
	}

	Mat image;
	image = imread(imageName, IMREAD_COLOR); // Read the file

	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		throw exception("Could not open or find the image");
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
}


void Camread() {
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		throw exception("Cannot open camera");

	cout << "stop by pressing ESC" << endl;

	Mat edges;
	namedWindow("edges", 1);

	while(waitKey(10) != 27)
	{
		Mat frame;
		cap >> frame;
		//if (frame.empty()) break; // end of video stream
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);

		imshow("this is you, smile! :)", frame);
		imshow("edges", edges);

	}
	// the camera will be closed automatically upon exit
	// cap.close();
	waitKey(0);
}


//void aprendizaje() {
//	CvMemStorage* storage = new CvMemStorage;
//
//}

/** Image processing, Face detection
* Detección de cuerpos a partir de un XML de referencia
* 
*/

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String body_cascade_name = "haarcascade_upperbody.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier body_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE , Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);

		Point p1(faces[i].x, faces[i].y);
		Point p2(faces[i].x + faces[i].width, faces[i].y+faces[i].height);
		
		//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(frame, p1, p2, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}

	//Detect upperbody
	vector<Rect> upperbody;

	body_cascade.detectMultiScale(frame_gray, upperbody, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < upperbody.size(); i++) {
		
		Point center(upperbody[i].x + upperbody[i].width*0.5, upperbody[i].y + upperbody[i].height*0.5);
		ellipse(frame, center, Size(upperbody[i].width*0.5, upperbody[i].height*0.5), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
	}


	//-- Show what you got
	imshow(window_name, frame);
}

int cameradetection(int argc, char** argv) {

	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!body_cascade.load(body_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	capture.open(0);
	if (capture.isOpened())
	{
		while (waitKey(10) != 27)
		{
			capture.read(frame);
			// cap >> frame;

			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

		}
		waitKey(0);
	}

}


int main(int argc, char** argv) {
	int opcion = 0;
	thread Image;
	thread Cam;
	
	try
	{
		do {
			menu(opcion);

			switch (opcion)
			{
			case 1:
				ReadImage(argc, argv);
				break;
			case 2:
				Camread();
				break;
			case 3:
				Image = thread(ReadImage, argc, argv);
				Cam = thread(Camread);
				Image.join();
				Cam.join();
				break;

			case 4: // Face detection
				cameradetection(argc, argv);
				break;

			default:
				cout << "bye bye" << endl;
				break;
			}
		} while (opcion != 0);

	}
	catch (const exception& e)
	{
		cout << "Se ha producido un error en la ejecución" << endl;
	}

	system("pause");
	return 0;
}