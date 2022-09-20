#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <iostream>
#include "readmaxT.h"
#include "readmidT.h"
#include <ostream>
#include <filesystem>

using std::filesystem::current_path;

float readmidT();
float readmaxT();


float a = readmaxT();
float b = readmidT();

TEST(corrtest,df0DFoam_max){
    EXPECT_FLOAT_EQ(a,2588.88);
}

TEST(corrtest,df0DFoam_mid){
    EXPECT_FLOAT_EQ(b,2007.72);
}


float readmaxT(){
    string str1 = get_current_dir_name();
    string str2 = "T";
    float a;
    string inFileName = "T" ;
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(100);
        while (inFile >> a){
       }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }
    cout << a << endl;
    return a;
}

float readmidT(){
    
    float a;
    float b;
    int i = 0;
    
    string inFileName = "T";
    ifstream inFile;
    inFile.open(inFileName.c_str());

    if (inFile.is_open())  
    {
        inFile.ignore(100);
        while (inFile >> a){
            i ++ ;
            if (i == 601){
                b = a;
            }
        }
    
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return b;
}