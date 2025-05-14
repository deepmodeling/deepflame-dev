#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <iostream>
#include <ostream>
#include <filesystem>
using namespace std;

float readmidTH2();
float readmaxTH2();

float readTGV(int k, string file);
float readHighSpeed();
float v = readHighSpeed();

// float H2maxT = readmaxTH2();
// float H2midT = readmidTH2();


float TGV500  = readTGV(806,"2DTGV/5/data_T.xy");
float TGV100 = readTGV(1100,"2DTGV/1/data_T.xy");
float TGV200 = readTGV(1064,"2DTGV/2/data_T.xy");
float TGV300 = readTGV(1064,"2DTGV/3/data_T.xy");
float TGV400 = readTGV(1098,"2DTGV/4/data_T.xy");


float readSandia(int k, string file);
float T1 = readSandia(1,"2DSandia/data_T.xy");
float T2 = readSandia(2,"2DSandia/data_T.xy");
float T3 = readSandia(3,"2DSandia/data_T.xy");
float T4 = readSandia(4,"2DSandia/data_T.xy");
float T5 = readSandia(5,"2DSandia/data_T.xy");
float T6 = readSandia(6,"2DSandia/data_T.xy");
float T7 = readSandia(7,"2DSandia/data_T.xy");
float T8 = readSandia(8,"2DSandia/data_T.xy");
float T9 = readSandia(9,"2DSandia/data_T.xy");
float T10 = readSandia(10,"2DSandia/data_T.xy");
float T11 = readSandia(11,"2DSandia/data_T.xy");

float readBomb(int k, string file);
float aachenBomb1  = readBomb(158,"aachenBomb2D/data_T.xy");
float aachenBomb2 =  readBomb(168,"aachenBomb2D/data_T.xy");
float aachenBomb3 =  readBomb(174,"aachenBomb2D/data_T.xy");
float aachenBomb4 =  readBomb(192,"aachenBomb2D/data_T.xy");


TEST(corrtest,dfHighSpeedFoam){
   EXPECT_NEAR(v,1979.33,19.79); // within 1% of the theroetical value
}

TEST(corrtest,dfLowMachFoam_TGV){
    EXPECT_FLOAT_EQ(TGV500,1535.03);   // compare the maximum temperature along y direction in 2D TGV after 500 time steps
    EXPECT_FLOAT_EQ(TGV400,1298.64);   //  ..........400 time steps
    EXPECT_FLOAT_EQ(TGV300,874.13098);
    EXPECT_FLOAT_EQ(TGV200,538.34802);
    EXPECT_FLOAT_EQ(TGV100,363.06601);
}

TEST(corrtest,2DSandia){
    EXPECT_FLOAT_EQ(T1,307.93594);
    EXPECT_FLOAT_EQ(T2,311.34987);
    EXPECT_FLOAT_EQ(T3,378.77716);
    EXPECT_FLOAT_EQ(T4,658.02573);
    EXPECT_FLOAT_EQ(T5,1106.6115);
    EXPECT_FLOAT_EQ(T6,1543.0465);
    EXPECT_FLOAT_EQ(T7,1888.2342);
    EXPECT_FLOAT_EQ(T8,1983.3421);
    EXPECT_FLOAT_EQ(T9,1797.2623);
    EXPECT_FLOAT_EQ(T10,1488.0969);
    EXPECT_FLOAT_EQ(T11,1081.8983);
}

TEST(corrtest,dfLowMachFoam_2DaachenBomb){
    EXPECT_NEAR(aachenBomb1,809.39398193359375,0.0001);
    EXPECT_NEAR(aachenBomb2,1814.1099853515625,0.0001);
    EXPECT_NEAR(aachenBomb3,912.61602783203125,0.0001);
    EXPECT_NEAR(aachenBomb4,2492.31005859375,0.0001);
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




float readSandia(int k, string file){

    float T,x;
    float b;
    int i = 0;

    string inFileName = file;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())
    {
        while (inFile >> x >> T){
            i ++ ;
            if (i == k){
                b = T;
            }
        }

    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}

float readBomb(int k, string file){

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
            if (i == k){
                b = a;
            }
        }

    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    cout << b << endl;

    return b;
}

float readflameSpeed(int k, string file);
float fs = readflameSpeed(3,"flameSpeed/fs");


TEST(corrtest,flameSpeed){
    EXPECT_FLOAT_EQ(fs,6);   // compare the maximum temperature along y direction in 2D TGV after 500 time steps
}

float readflameSpeed(int k, string file){

    float fs;
    float b;
    int i = 0;

    string inFileName = file;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())
    {
        while (inFile >> fs){
            i ++ ;
            if (i == k){
                b = fs;
            }
        }

    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}
