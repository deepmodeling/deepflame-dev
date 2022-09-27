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
float readTGV();


float H2maxT = readmaxTH2();
float H2midT = readmidTH2();
float CH4maxT = readmaxTCH4();
float CH4midT = readmidTCH4();
float TGVmin  = readTGV();


TEST(corrtest,df0DFoam_H2){
    EXPECT_FLOAT_EQ(H2maxT,2588.88);   // compare the maximum temperature of H2 case 
    EXPECT_FLOAT_EQ(H2midT,1298.12); // compare the temperature of H2 case at the maximum gradient when t = 0.000245s
}

TEST(corrtest,df0DFoam_CH4){
    EXPECT_FLOAT_EQ(CH4maxT,2816.82);   // compare the maximum temperature of CH4 case 
    EXPECT_FLOAT_EQ(CH4midT,2410.39); // compare the temperature of CH4 case at the maximum gradient when t = 0.000249s
}

TEST(corrtest,dfLowMachFoam_TGV){
    EXPECT_FLOAT_EQ(TGVmin,372.156);   // compare the maximum temperature of CH4 case 
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


float readTGV(){
    
    float a;
    float b;
    int i = 0;
    
    string inFileName = "2DTGV/data_T.xy";
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        while (inFile >> a){
            i ++ ;
            if (i == 1058 ){  // minimum temperature
                b = a;
            }
        }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}
