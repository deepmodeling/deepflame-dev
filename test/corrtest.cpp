#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <iostream>
#include <ostream>
#include <filesystem>
using namespace std;

float readmidTH2();
float readmaxTH2();
float readmidTCH4();
float readmaxTCH4();
float readTGV(int k, string file);
float readHighSpeed();
float v = readHighSpeed();

float H2maxT = readmaxTH2();
float H2midT = readmidTH2();
float CH4maxT = readmaxTCH4();
float CH4midT = readmidTCH4();
float TGV500  = readTGV(806,"2DTGV/5/data_T.xy");
float TGV100 = readTGV(1100,"2DTGV/1/data_T.xy");
float TGV200 = readTGV(1064,"2DTGV/2/data_T.xy");
float TGV300 = readTGV(1064,"2DTGV/3/data_T.xy");
float TGV400 = readTGV(1098,"2DTGV/4/data_T.xy");


TEST(corrtest,df0DFoam_H2){
    EXPECT_FLOAT_EQ(H2maxT,2588.88);   // compare the maximum temperature of H2 case 
    EXPECT_FLOAT_EQ(H2midT,1298.12); // compare the temperature of H2 case at the maximum gradient when t = 0.000245s
}

TEST(corrtest,df0DFoam_CH4){
    EXPECT_FLOAT_EQ(CH4maxT,2816.82);   // compare the maximum temperature of CH4 case 
    EXPECT_FLOAT_EQ(CH4midT,2410.39); // compare the temperature of CH4 case at the maximum gradient when t = 0.000249s
}

TEST(corrtest,dfLowMachFoam_TGV){
    EXPECT_FLOAT_EQ(TGV500,1534.77);   // compare the maximum temperature along y direction in 2D TGV after 500 time steps
    EXPECT_FLOAT_EQ(TGV400,1313.03);   //  ..........400 time steps
    EXPECT_FLOAT_EQ(TGV300,887.622);
    EXPECT_FLOAT_EQ(TGV200,553.332);
    EXPECT_FLOAT_EQ(TGV100,364.106);
}

TEST(corrtest,dfHighSpeedFoam){
    EXPECT_NEAR(v,1979.33,19.79); // within 1% of the theroetical value
}



float readmaxTH2(){
    float a;
    string inFileName = "0DH2/T" ;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(80);
        while (inFile >> a){
       }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }
    cout << a << endl;
    return a;
}

float readmaxTCH4(){
    float a;
    string inFileName = "0DCH4/T" ;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(80);
        while (inFile >> a){
       }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }
    cout << a << endl;
    return a;
}

float readmidTH2(){
    
    float a;
    float b;
    int i = 0;
    
    string inFileName = "0DH2/T";
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(80);
        while (inFile >> a){
            i ++ ;
            if (i == 490 ){  // t = 0.000245 dt = 37.25, maximum gradient
                b = a;
            }
        }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}


float readmidTCH4(){
    
    float a;
    float b;
    int i = 0;
    
    string inFileName = "0DCH4/T";
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(80);
        while (inFile >> a){
            i ++ ;
            if (i == 498 ){  // t = 0.000249 dt = 84.165, maximum gradient
                b = a;
            }
        }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}


float readTGV(int k, string file){
    
    float a;
    float b;
    int i = 0;
    
    string inFileName = file;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        while (inFile >> a){
            i ++ ;
            if (i == k){  // minimum temperature
                b = a;
            }
        }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}


float readHighSpeed(){
    float xsum=0,x2sum=0,ysum=0,xysum=0;
    float t;
    char dummy;
    char p;
    float minp;
    float minloc;
    int processor;
    float max;
    float maxloc;
    float maxloc_x;
    int processor2;
    float slope;
    int i = 0;
    float slope2;

    string inFileName = "1Ddetonation/fieldMinMax.dat" ;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(162);
        while(inFile >> t >> p >> minp >> dummy >> minloc>> minloc >> minloc >> dummy >> processor >> max >> dummy >> maxloc_x >> maxloc >> maxloc >> dummy >> processor){
            i = i +1;
            if (i >= 30){
                xsum = xsum+t;
                ysum = ysum+ maxloc_x;
                x2sum = x2sum + t * t;
                xysum = xysum + t*maxloc_x;
            }
        };
        //while (inFile >> t >> p >> minp >> minlocation >> processor >> max >> maxlocation >> processor2){
       //} 
        slope = (15*xysum-xsum*ysum)/(15*x2sum-xsum*xsum);

    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }
    
    
    return slope;
}
