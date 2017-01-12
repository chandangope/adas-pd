
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

int main(int argc, char* argv[])
{

#if 0
  	writeImageData(argv[1]);
#endif

#if 1
	cout << "*****************************************************" << "\n";
	cout << "ivadaspd testapp: Classify Data" << "\n";
	cout << "*****************************************************" << "\n";
	if(argc < 2){
		cout << "Number of params should be at least 1" << "\n";
		return 1;
	}
	string modelFilePath = "../model/frozen_graph.pb";
	void* pdEngine = ivAdasPDEngine_Init(modelFilePath);

	auto t1 = Clock::now();
	for(int i=1; i<argc; i++){
		cout << endl << "---------------------------------------------------" << "\n";
		cout <<  "Processing " << argv[i] << std::endl;
		classifyImageData(pdEngine, argv[i]);
	}
	auto t2 = Clock::now();
	cout << endl << "---------------------------------------------------" << "\n";
	std::cout << "Total time taken: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-6
              << " milliseconds" << std::endl;
	std::cout << "Average time taken: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-6/(argc-1)
              << " milliseconds" << std::endl;
 	
 	cout << endl << "*****************************************************" << "\n";
 	if(ivAdasPDEngine_Close(pdEngine) == 0)
  		cout <<  "Session closed" << std::endl;
  	else
  		cout <<  "Session closed failed" << std::endl;
  	cout << "*****************************************************" << "\n";
#endif

}

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

#if 0
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

  Size size(36,72); //(cols,rows)
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