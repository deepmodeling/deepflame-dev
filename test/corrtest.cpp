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


float H2maxT = readmaxTH2();
float H2midT = readmidTH2();
float CH4maxT = readmaxTCH4();
float CH4midT = readmidTCH4();
float TGV500  = readTGV(806,"2DTGV/5/data_T.xy");
float TGV100 = readTGV(1098,"2DTGV/1/data_T.xy");
float TGV200 = readTGV(1016,"2DTGV/2/data_T.xy");
float TGV300 = readTGV(1048,"2DTGV/3/data_T.xy");
float TGV400 = readTGV(1048,"2DTGV/4/data_T.xy");


TEST(corrtest,df0DFoam_H2){
    EXPECT_FLOAT_EQ(H2maxT,2588.88);   // compare the maximum temperature of H2 case 
    EXPECT_FLOAT_EQ(H2midT,1298.12); // compare the temperature of H2 case at the maximum gradient when t = 0.000245s
}

TEST(corrtest,df0DFoam_CH4){
    EXPECT_FLOAT_EQ(CH4maxT,2816.82);   // compare the maximum temperature of CH4 case 
    EXPECT_FLOAT_EQ(CH4midT,2410.39); // compare the temperature of CH4 case at the maximum gradient when t = 0.000249s
}

TEST(corrtest,dfLowMachFoam_TGV){
    EXPECT_FLOAT_EQ(TGV500,1533.31);   // compare the maximum temperature along y direction in 2D TGV after 500 time steps
    EXPECT_FLOAT_EQ(TGV400,1303.71);   //  ..........400 time steps
    EXPECT_FLOAT_EQ(TGV300,880.402);
    EXPECT_FLOAT_EQ(TGV200,546.348);
    EXPECT_FLOAT_EQ(TGV100,364.29);
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


