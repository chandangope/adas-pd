
#include "iv_adas_pd.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

typedef std::chrono::high_resolution_clock Clock;


int writeImageData(char* imgPath);
int classifyImageData(void* pdEngine, char* imgDataPath);
int detectInImageData(void* pdEngine, char* imgDataPath);
int displayImageData(char* imgPath);
int resize_nn(const unsigned char* imgDataIn, int widthIn, int heightIn,
              unsigned char* imgDataOut, int widthOut, int heightOut);
int detectInVideo(char* videoPath);
int getROIBoxes(ivAdasPDBbox* roiBoxes, int& numBoxes, ivAdasPDBbox* roiWindow);


/*./testivadaspd -vf http://192.168.1.166:4747/mjpegfeed?1280x720*/

/*#define WRITE_DEMOVIDEO*/


int main(int argc, char* argv[])
{
  if(argc < 2){
    cout << "Number of params should be at least 1" << "\n";
    return 1;
  }
  if(strcmp(argv[1], "-vf")==0){
    detectInVideo(argv[2]);
  }

#if 0
  	writeImageData(argv[1]);
    //displayImageData(argv[1]);

    /*ifstream fin(argv[1], ios::binary);
    if(fin.is_open()){
      // get its size:
      fin.seekg(0, ios::end);
      int fileSize = fin.tellg();
      cout <<  "Data size in bytes = " << fileSize << endl;
      if(fileSize != 48*96*sizeof(unsigned char)){
        fin.close();
        cout <<  "Unexpected Image data size" << std::endl;
        return -1;
      }
      fin.seekg(0, ios::beg);

      std::vector<unsigned char> imagevals(48*96);
      fin.read(reinterpret_cast<char*>(&imagevals[0]), imagevals.size()*sizeof(unsigned char));
      fin.close();
      cout <<  "Image data read" << std::endl;
      int heightOut = 72; int widthOut = 36;
      unsigned char* imgDataOut = new unsigned char[heightOut*widthOut];
      resize_nn(&imagevals[0], 48, 96, imgDataOut, widthOut, heightOut);
      delete[] imgDataOut;
    }*/
#endif

#if 0
	cout << "*****************************************************" << "\n";
	cout << "ivadaspd testapp: Detect in image" << "\n";
	cout << "*****************************************************" << "\n";
	if(argc < 2){
		cout << "Number of params should be at least 1" << "\n";
		return 1;
	}
	string modelFilePath = "../model/frozen_graph.pb";
	void* pdEngine = ivAdasPDEngine_Init(modelFilePath);

  auto totalT = 0;
	for(int i=1; i<argc; i++){
		cout << endl << "---------------------------------------------------" << "\n";
		cout <<  "Processing " << argv[i] << std::endl;
		//classifyImageData(pdEngine, argv[i]);
    auto t1 = Clock::now();
    detectInImageData(pdEngine, argv[i]);
    auto t2 = Clock::now();
    totalT += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-6;
    cout << endl << "---------------------------------------------------" << "\n";
    std::cout << "Time taken: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-6
              << " milliseconds" << std::endl;
	}
  cout << endl << endl << "*****************************************************" << "\n";
  std::cout << "Average Time taken: " 
              << totalT/(argc-1)
              << " milliseconds" << std::endl;
 	cout << endl << "*****************************************************" << "\n";
 	if(ivAdasPDEngine_Close(pdEngine) == 0)
  		cout <<  "Session closed" << std::endl;
  	else
  		cout <<  "Session closed failed" << std::endl;
  	cout << "*****************************************************" << "\n";
#endif
}

int detectInVideo(char* videoPath)
{
  cout << "*****************************************************" << "\n";
  cout << "ivadaspd testapp: Detect in Video" << "\n";
  cout << "*****************************************************" << "\n";

  string modelFilePath = "../model/frozen_graph.pb";
  void* pdEngine = ivAdasPDEngine_Init(modelFilePath);

  cv::VideoCapture inputVideo;
  if(videoPath == NULL){
    inputVideo = VideoCapture(0);
  }
  else
    inputVideo = VideoCapture(videoPath);

  if (!inputVideo.isOpened())
  {
    std::cout << "Could not open the input video: " << videoPath << std::endl;
    return -1;
  }

  cv::Size S = cv::Size((int)inputVideo.get(CV_CAP_PROP_FRAME_WIDTH), (int)inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
  std::cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height << std::endl;

  cv::Size S_resized = S;
  float resizeY = 1.f;
  float resizeX = 1.f;
  float resizedHeight = 720;
  float resizedWidth = 1280;
  if (S.height > resizedHeight)
  {
    resizeY = S.height / resizedHeight;
    resizeX = S.width / resizedWidth;
    S_resized.height = S.height / resizeY;
    S_resized.width = S.width / resizeX;
  }
  std::cout << "Processing frame resolution: Width=" << S_resized.width << "  Height=" << S_resized.height << std::endl;

  #ifdef WRITE_DEMOVIDEO
    string NAME = "demopd.avi";   // Form the new name with container
    int fourcc = CV_FOURCC('X', 'V', 'I', 'D');
    cv::Size S_out = cv::Size(resizedWidth, resizedHeight);
    cv::VideoWriter output_cap(NAME, fourcc, 20, S_out);
    if (!output_cap.isOpened())
    {
      std::cout << "!!! Output video could not be opened" << std::endl;
      return -1;
    }
  #endif

  cv::Mat srcOrig;
  cv::Mat src(S_resized, srcOrig.type());
  int D = 10;
  int delay = D;
  int conf;
  char text[256];
  int numboxes = 0;
  ivAdasPDBbox roiBoxes[MAX_IVADASPD_BBOXES];
  ivAdasPDBbox roiWindow;
  int frameCnt = 0;
  auto totalT = 0;
  float fps = 0;
  for (;;) //Show the image captured in the window and repeat
  {
    inputVideo >> srcOrig;    
    if (srcOrig.empty()) break;
    frameCnt++;
    if(frameCnt == 150)
    {
      fps = 1000.f/(totalT/frameCnt);
      frameCnt = 0;
      totalT = 0;
    }
    cv::resize(srcOrig, src, S_resized);
    vector<unsigned char> imagevals;
    {
      for(int r=0; r<resizedHeight; r++){
        for(int c=0; c<resizedWidth; c++){
          Vec3b vals = src.at<Vec3b>(r,c);
          uchar blue = vals.val[0];
          uchar green = vals.val[1];
          uchar red = vals.val[2];
          unsigned char intensity = ((blue+green+red)/3.f);
          imagevals.push_back(intensity);
        }
      }
    }
    auto t1 = Clock::now();
    int res = ivAdasPDEngine_detectInImage(pdEngine, reinterpret_cast<unsigned char*>(&imagevals[0]), resizedHeight, resizedWidth,
      &roiBoxes[0], &numboxes, &roiWindow);
    auto t2 = Clock::now();
    totalT += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-6;
    imagevals.clear();

    cv::Rect roiWindowRect = cv::Rect(roiWindow.topLeftX, roiWindow.topLeftY, roiWindow.width, roiWindow.height);
    cv::rectangle(src, roiWindowRect, CV_RGB(0, 255, 0), 4);
    for(int n=0; n<numboxes; n++){
      cv::Rect roiRect = cv::Rect(roiBoxes[n].topLeftX, roiBoxes[n].topLeftY, roiBoxes[n].width, roiBoxes[n].height);
      cv::rectangle(src, roiRect, CV_RGB(0, 0, 255), 1);
    }

    if(res == 0){
      for(int n=0; n<numboxes; n++){
        cv::Rect detectedRect = cv::Rect(roiBoxes[n].topLeftX, roiBoxes[n].topLeftY, roiBoxes[n].width, roiBoxes[n].height);
        if(roiBoxes[n].detectionConf > 0.95){
          cv::rectangle(src, detectedRect, CV_RGB(255, 0, 0), 2);
        }
      }
    }
    else{
      cout << "ivAdasPDEngine_detectInImage() failed" << "\n";
    }
  #ifdef WRITE_DEMOVIDEO
      output_cap << src;
  #endif

    char txt[256];
    sprintf(txt, "fps=%d", (int)fps);
    putText(src, txt, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255, 255, 255), 2, 8);
    putText(src, txt, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0, 0, 0), 2, 8);
    imshow("Video", src);
    char e = cv::waitKey(delay);
    switch (e)
    {
    case 27: //ESC
      return 0;
      break;
    case 32: //Space
      delay = D - delay;
      break;
    default:
      break;
    }
  }
  #ifdef WRITE_DEMOVIDEO
    output_cap.release();
  #endif

    return 0;
}

int getROIBoxes(ivAdasPDBbox* roiBoxes, int& numBoxes, ivAdasPDBbox* roiWindow)
{
  numBoxes = 0;

  roiWindow->topLeftX = 100000;
  roiWindow->topLeftY = 100000;
  int roiWindow_bottomRightX = -1;
  int roiWindow_bottomRightY = -1;

  int marginX, topLeftX, topLeftY, width, height;

  /*marginX = 240;
  topLeftX = marginX;
  topLeftY = 180; //140
  width = 36;
  height = 72;
  while(true){
    if(topLeftX+width > 640-marginX) break;
    roiBoxes[numBoxes].width = width;
    roiBoxes[numBoxes].height = height;
    roiBoxes[numBoxes].topLeftY = topLeftY;
    roiBoxes[numBoxes].topLeftX = topLeftX;

    if(roiWindow->topLeftX > roiBoxes[numBoxes].topLeftX)
      roiWindow->topLeftX = roiBoxes[numBoxes].topLeftX;
    if(roiWindow->topLeftY > roiBoxes[numBoxes].topLeftY)
      roiWindow->topLeftY = roiBoxes[numBoxes].topLeftY;
    if(roiWindow_bottomRightX < roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width)
      roiWindow_bottomRightX = roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width;
    if(roiWindow_bottomRightY < roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height)
      roiWindow_bottomRightY = roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height;

    numBoxes++;
    topLeftX += width/2;
  }*/

  /*marginX = 260;
  topLeftX = marginX;
  topLeftY = 150; //130
  width = 48;
  height = 96;
  while(true){
    if(topLeftX+width > 640-marginX) break;
    roiBoxes[numBoxes].width = width;
    roiBoxes[numBoxes].height = height;
    roiBoxes[numBoxes].topLeftY = topLeftY;
    roiBoxes[numBoxes].topLeftX = topLeftX;

    if(roiWindow->topLeftX > roiBoxes[numBoxes].topLeftX)
      roiWindow->topLeftX = roiBoxes[numBoxes].topLeftX;
    if(roiWindow->topLeftY > roiBoxes[numBoxes].topLeftY)
      roiWindow->topLeftY = roiBoxes[numBoxes].topLeftY;
    if(roiWindow_bottomRightX < roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width)
      roiWindow_bottomRightX = roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width;
    if(roiWindow_bottomRightY < roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height)
      roiWindow_bottomRightY = roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height;

    numBoxes++;
    topLeftX += width/3;
  }*/

  marginX = 200;
  topLeftX = marginX;
  topLeftY = 140; //130
  width = 64;
  height = 128;
  while(true){
    if(topLeftX+width > 640-marginX) break;
    roiBoxes[numBoxes].width = width;
    roiBoxes[numBoxes].height = height;
    roiBoxes[numBoxes].topLeftY = topLeftY;
    roiBoxes[numBoxes].topLeftX = topLeftX;

    if(roiWindow->topLeftX > roiBoxes[numBoxes].topLeftX)
      roiWindow->topLeftX = roiBoxes[numBoxes].topLeftX;
    if(roiWindow->topLeftY > roiBoxes[numBoxes].topLeftY)
      roiWindow->topLeftY = roiBoxes[numBoxes].topLeftY;
    if(roiWindow_bottomRightX < roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width)
      roiWindow_bottomRightX = roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width;
    if(roiWindow_bottomRightY < roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height)
      roiWindow_bottomRightY = roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height;

    numBoxes++;
    topLeftX += width/3;
  }

  marginX = 200;
  topLeftX = marginX;
  topLeftY = 100;
  width = 96;
  height = 192;
  while(true){
    if(topLeftX+width > 640-marginX) break;
    roiBoxes[numBoxes].width = width;
    roiBoxes[numBoxes].height = height;
    roiBoxes[numBoxes].topLeftY = topLeftY;
    roiBoxes[numBoxes].topLeftX = topLeftX;

    if(roiWindow->topLeftX > roiBoxes[numBoxes].topLeftX)
      roiWindow->topLeftX = roiBoxes[numBoxes].topLeftX;
    if(roiWindow->topLeftY > roiBoxes[numBoxes].topLeftY)
      roiWindow->topLeftY = roiBoxes[numBoxes].topLeftY;
    if(roiWindow_bottomRightX < roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width)
      roiWindow_bottomRightX = roiBoxes[numBoxes].topLeftX + roiBoxes[numBoxes].width;
    if(roiWindow_bottomRightY < roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height)
      roiWindow_bottomRightY = roiBoxes[numBoxes].topLeftY + roiBoxes[numBoxes].height;

    numBoxes++;
    topLeftX += width/2;
  }


  roiWindow->width = roiWindow_bottomRightX - roiWindow->topLeftX;
  roiWindow->height = roiWindow_bottomRightY - roiWindow->topLeftY;
  roiWindow->detectionConf = -1;

  return 0;
}

#if 1
int detectInImageData(void* pdEngine, char* imgDataPath)
{
  ifstream fin(imgDataPath, ios::binary);
  if(fin.is_open()){
    // get its size:
    fin.seekg(0, ios::end);
    int fileSize = fin.tellg();
    cout <<  "Data size in bytes = " << fileSize << endl;
    if(fileSize != 640*360*sizeof(unsigned char)){
      fin.close();
      cout <<  "Unexpected Image data size" << std::endl;
      return -1;
    }
    fin.seekg(0, ios::beg);

    std::vector<unsigned char> imagevals(640*360);
    fin.read(reinterpret_cast<char*>(&imagevals[0]), imagevals.size()*sizeof(unsigned char));
    fin.close();
    cout <<  "Image data read" << std::endl;

    ivAdasPDBbox detectedBoxes[MAX_IVADASPD_BBOXES];
    int numDetectedBoxes = 0;
    /*float detectedConfs[MAX_IVADASPD_BBOXES];
    int res = ivAdasPDEngine_detectInImage(pdEngine, reinterpret_cast<unsigned char*>(&imagevals[0]),360, 640,
      &detectedBoxes[0], &numDetectedBoxes, &detectedConfs[0]);*/

    int marginX = 200;
    int topLeftX = marginX;
    int topLeftY = 130;
    int width = 64;
    int height = 128;
    int maxBoxes = 12;
    for(int i=0; i<maxBoxes; i++){
      if(topLeftX > 640-marginX) break;
      detectedBoxes[numDetectedBoxes].width = width;
      detectedBoxes[numDetectedBoxes].height = height;
      detectedBoxes[numDetectedBoxes].topLeftY = topLeftY;
      detectedBoxes[numDetectedBoxes].topLeftX = topLeftX;
      cout << "x=" << detectedBoxes[numDetectedBoxes].topLeftX << ",";
      numDetectedBoxes++;
      topLeftX += width/2;
    }
    cout << endl;
    cout << endl;

    marginX = 200;
    topLeftX = marginX;
    topLeftY = 120;
    width = 36;
    height = 72;
    maxBoxes = 14;
    for(int i=0; i<maxBoxes; i++){
      if(topLeftX > 640-marginX) break;
      detectedBoxes[numDetectedBoxes].width = width;
      detectedBoxes[numDetectedBoxes].height = height;
      detectedBoxes[numDetectedBoxes].topLeftY = topLeftY;
      detectedBoxes[numDetectedBoxes].topLeftX = topLeftX;
      cout << "x=" << detectedBoxes[numDetectedBoxes].topLeftX << ",";
      numDetectedBoxes++;
      topLeftX += width/2;
    }
    cout << endl;
    cout << endl;

    /*topLeftX = 140;
    topLeftY = 130;
    width = 48;
    height = 96;
    maxBoxes = 20;
    for(int i=0; i<maxBoxes; i++){
      detectedBoxes[numDetectedBoxes].width = width;
      detectedBoxes[numDetectedBoxes].height = height;
      detectedBoxes[numDetectedBoxes].topLeftY = topLeftY;
      detectedBoxes[numDetectedBoxes].topLeftX = topLeftX;
      cout << "x=" << detectedBoxes[numDetectedBoxes].topLeftX << ",";
      numDetectedBoxes++;
      topLeftX += width/3;
    }
    cout << endl;
    cout << endl;
    
    topLeftX = 100;
    topLeftY = 120;
    width = 64;
    height = 128;
    maxBoxes = 18;
    for(int i=0; i<maxBoxes; i++){
      detectedBoxes[numDetectedBoxes].width = width;
      detectedBoxes[numDetectedBoxes].height = height;
      detectedBoxes[numDetectedBoxes].topLeftY = topLeftY;
      detectedBoxes[numDetectedBoxes].topLeftX = topLeftX;
      cout << "x=" << detectedBoxes[numDetectedBoxes].topLeftX << ",";
      numDetectedBoxes++;
      topLeftX += width/3;
    }
    cout << endl;*/

    int res = ivAdasPDEngine_classifyBboxesInImage(pdEngine, reinterpret_cast<unsigned char*>(&imagevals[0]),360, 640,
      &detectedBoxes[0], numDetectedBoxes);
    imagevals.clear();

    if(res == 0){
      cout << endl << "Detected boxes:" << "\n";
      for(int n=0; n<numDetectedBoxes; n++){
        if(detectedBoxes[n].detectionConf > 0.8){
          cout << "conf=" << detectedBoxes[n].detectionConf;
          cout << ", x=" << detectedBoxes[n].topLeftX << ", y=" << detectedBoxes[n].topLeftY;
          cout << ", width=" << detectedBoxes[n].width << ", height=" << detectedBoxes[n].height << "\n";

        }
      }
    }
    else{
      cout << "ivAdasPDEngine_detectInImage() failed" << "\n";
    }
  }
  else{
    cout <<  "Could not read imagedata" << std::endl;
    return -1;
  }
  return 0;
}
#endif




#if 1
int classifyImageData(void* pdEngine, char* imgDataPath)
{
	ifstream fin(imgDataPath, ios::binary);
	if(fin.is_open())
  	{
  		// get its size:
    	fin.seekg(0, ios::end);
    	int fileSize = fin.tellg();
    	cout <<  "Data size in bytes = " << fileSize << endl;
    	if(fileSize != 36*72*sizeof(unsigned char)){
    		fin.close();
  			cout <<  "Unexpected Image data size" << std::endl;
  			return -1;
    	}
    	fin.seekg(0, ios::beg);

  		std::vector<unsigned char> imagevals(36*72);
  		fin.read(reinterpret_cast<char*>(&imagevals[0]), imagevals.size()*sizeof(unsigned char));
  		fin.close();
  		cout <<  "Image data read" << std::endl;

  		int classOut;float confOut;
		/*std::vector<float> pixelData;
		for(int k=0; k<28*28; k++){
			pixelData.push_back(k);
		}*/
		//int res = classifyimage_bbox(reinterpret_cast<unsigned char*>(&imagevals[0]), 28, 28, &classOut, &confOut);
		int res = ivAdasPDEngine_classifyimage_bbox(pdEngine,
			reinterpret_cast<unsigned char*>(&imagevals[0]), 72, 36, &classOut, &confOut);
		if(res == 0)
			cout << "class=" << classOut << " conf=" << confOut << "\n";
		else
			cout << "ivAdasPDEngine_classifyimage_bbox() failed" << "\n";

  	}
  	else{
  		cout <<  "Could not read imagedata" << std::endl;
  		return -1;
  	}

	return 0;
}
#endif

#if 1
int displayImageData(char* imgPath)
{
  ifstream fin(imgPath, ios::binary);
  if(fin.is_open())
    {
      // get its size:
      fin.seekg(0, ios::end);
      int fileSize = fin.tellg();
      cout <<  "Data size in bytes = " << fileSize << endl;
      if(fileSize != 36*72*sizeof(unsigned char)){
        fin.close();
        cout <<  "Unexpected Image data size" << std::endl;
        return -1;
      }
      fin.seekg(0, ios::beg);

      std::vector<unsigned char> imagevals(36*72);
      //unsigned char* image_uchar = new unsigned char[36*72];
      fin.read(reinterpret_cast<char*>(&imagevals[0]), imagevals.size()*sizeof(unsigned char));
      fin.close();
      for(int i=0; i<36*72; i++){
        cout <<  (int)imagevals[i] << ", ";
      }
      cout <<  "Image data read" << std::endl;
      Mat img(72,36,CV_8UC1,&imagevals[0],cv::Mat::AUTO_STEP);
      namedWindow( "Image", WINDOW_AUTOSIZE );// Create a window for display.
      imshow( "Image", img);                   // Show our image inside it.
      waitKey(0);
    }
    else{
      cout <<  "Could not read imagedata" << std::endl;
      return -1;
    }

  return 0;
}
#endif

#if 1
int writeImageData(char* imgPath)
{
	// Initialize a tensorflow session
  cout << "ivadaspd testapp: Write Data" << "\n";

	Mat image;
  image = imread(imgPath, CV_LOAD_IMAGE_COLOR);   // Read the file
  if(! image.data )                              // Check for invalid input
  {
  	cout <<  "Could not open or find the image" << std::endl;
  	return -1;
  }

  //Size size(36,72); //(cols,rows)
  //Size size(48,96); //(cols,rows)
  Size size(640,360); //(cols,rows)
  Mat dst;//dst image
  resize(image,dst,size);//resize image
  namedWindow( "Image", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Image", dst );                   // Show our image inside it.
  waitKey(0);

  Size sz = dst.size();
  vector<unsigned char> imagevals;
  ofstream fout("imagedata.bin", ios::binary);
  if(fout.is_open())
  {
  	for(int r=0; r<sz.height; r++){
  		for(int c=0; c<sz.width; c++){
  			Vec3b vals = dst.at<Vec3b>(r,c);
  			uchar blue = vals.val[0];
  			uchar green = vals.val[1];
  			uchar red = vals.val[2];
  			unsigned char intensity = ((blue+green+red)/3.f);
  			imagevals.push_back(intensity);
  		}
  	}
  	fout.write(reinterpret_cast<char*>(&imagevals[0]), imagevals.size()*sizeof(unsigned char));
  	fout.close();
  	cout <<  "Image data written to imagedata.bin" << std::endl;
  }
  else{
  	cout <<  "Could not open imagedata.bin for writing" << std::endl;
  	return -1;
  }
}
#endif


#if 0
int resize_nn(const unsigned char* imgDataIn, int widthIn, int heightIn,
              unsigned char* imgDataOut, int widthOut, int heightOut)
{
  unsigned char* ptrOut = imgDataOut;
  float x_scaleFactor = (widthIn-1)/(float)widthOut;
  float y_scaleFactor = (heightIn-1)/(float)heightOut;
  //cout << "x_scaleFactor, y_scaleFactor=" << x_scaleFactor << ", " << y_scaleFactor << std::endl;

  for(int row = 0; row < heightOut; row++){
    for(int col = 0; col < widthOut; col++){
      int xIn = (int)(x_scaleFactor*col);
      int yIn = (int)(y_scaleFactor*row);
      int offset = yIn*widthIn + xIn;

      /*float x_diff = (x_scaleFactor * col) - xIn;
      float y_diff = (y_scaleFactor * row) - yIn;
      int A = *(imgDataIn + offset);
      int B = *(imgDataIn + offset + 1);
      int C = *(imgDataIn + offset + widthIn);
      int D = *(imgDataIn + offset + widthIn + 1);
      *ptrOut = A*(1-x_diff)*(1-y_diff) + B*(x_diff)*(1-y_diff) + C*(y_diff)*(1-x_diff) + D*(x_diff*y_diff);*/

      *ptrOut = *(imgDataIn + offset);
      ptrOut++;
    }
  }

  Mat img(heightOut,widthOut,CV_8UC1,imgDataOut,cv::Mat::AUTO_STEP);
  namedWindow( "Resized" );// Create a window for display.
  imshow( "Resized", img);                   // Show our image inside it.
  waitKey(0);

  ofstream fout("imagedata.bin", ios::binary);
  if(fout.is_open())
  {
    fout.write(reinterpret_cast<char*>(imgDataOut), widthOut*heightOut*sizeof(unsigned char));
    fout.close();
    cout <<  "Image data written to imagedata.bin" << std::endl;
  }

  return 0;
}
#endif
