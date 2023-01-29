    Info<< "Reading " << H_fuel << "\n" << endl;
    Info<< "Reading " << H_ox << "\n" << endl;

    z_Tb5={new double[NZL]{}};
    sl_Tb5={new double[NZL]{}};
    th_Tb5={new double[NZL]{}};
    tau_Tb5={new double[NZL]{}};
    kctau_Tb5={new double[NZL]{}};

    Info<< "Reading laminar flame properties\n" << endl;

    for(int jj = 0; jj < NZL; jj++)  
    {
         f1=fscanf( table, "%lf %lf %lf %lf %lf\n",
                &z_Tb5[jj],&sl_Tb5[jj],&th_Tb5[jj],
                &tau_Tb5[jj],&kctau_Tb5[jj] );
    }

    fNZ=fscanf(table, "%d",&NZ);
    z_Tb3={ new double[NZ]{} };
    for(int ii = 0; ii < NZ; ii++)
    {
         f2=fscanf( table, "%lf\n", &z_Tb3[ii] );   //load z to z_Tb3
    }

    fNC=fscanf(table, "%d",&NC);
    c_Tb3={ new double[NC]{} };
    for(int ii = 0; ii < NC; ii++)  
    {
         f3=fscanf( table, "%lf\n", &c_Tb3[ii] );  ///load c to c_Tb3
    }

    fNGZ=fscanf(table, "%d",&NGZ);
    gz_Tb3={ new double[NGZ]{} };
    for(int ii = 0; ii < NGZ; ii++)   
    {
         f4=fscanf( table, "%lf\n", &gz_Tb3[ii] );  //load z-varance to gz_Tb3
    }

    fNGC=fscanf(table, "%d",&NGC);
    gc_Tb3={ new double[NGC]{} };
    for(int ii = 0; ii < NGC; ii++)
    {
         f5=fscanf( table, "%lf\n", &gc_Tb3[ii] );   //load c-variance to gc_Tb3
    }   

    fNZC=fscanf(table, "%d",&NZC);
    gzc_Tb3={ new double[NZC]{} };
    for(int ii = 0; ii < NZC; ii++)
    {
         f6=fscanf( table, "%lf\n", &gzc_Tb3[ii] );   //load zc-vovariance to gzc_Tb3
    }    

    fNS_NY=fscanf(table, "%d %d",&NS,&NY);

    omgc_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };     
    cOc_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };      
    ZOc_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };     
    cp_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };       
    hiyi_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };     
    mwt_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };      
    Tf_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };       
    nu_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };       
    Ycmax_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };   

    Yi01_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} },    
    Yi02_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} },    
    Yi03_Tb3={ new double[NZ*NC*NGZ*NGC*NZC]{} };    

    Info << "NS is read as: " << NS << endl;

    if(NS == 8)
    {
        scaledPV_ = true;
        Info<< "=============== Using scaled PV ==============="
            << "\n" << endl;
    }
    else if(NS == 9)
    {
        scaledPV_ = false;
        Info<< "=============== Using unscaled PV ==============="
            << "\n" << endl;
    }
    else
    {
        WarningInFunction << "Number of columns wrong in flare.tbl !!!"
                            << "\n" << endl;
    }

    const char* fmt8{"%lf %lf %lf %lf %lf %lf %lf %lf"};
    const char* fmt9{"%lf %lf %lf %lf %lf %lf %lf %lf %lf"};

    //Note：if nYis is changed，fmt_nYis is also needed to change！
    const char* fmt_nYis{"%lf %lf %lf"};


    Info<< "Reading turbulent flame properties\n" << endl;   

    int count = 0;    
    for(int ii = 0; ii < NZ; ii++)
    {
        for(int jj = 0; jj < NC; jj++)
        {
            for(int kk = 0; kk < NGZ; kk++)
            {
                for(int ll = 0; ll < NGC; ll++)
                {
                    for(int mm = 0; mm < NZC; mm++)
                    {
                        if(scaledPV_)  
                        {
                             f7=fscanf
                                     (
                                        table,fmt8,&omgc_Tb3[count],&cOc_Tb3[count],&ZOc_Tb3[count],&cp_Tb3[count],
                                        &mwt_Tb3[count],&hiyi_Tb3[count],&Tf_Tb3[count],&nu_Tb3[count]
                                     );   
                        }
                        else
                        {
                             f8=fscanf
                                     (
                                        table,fmt9,&omgc_Tb3[count],&cOc_Tb3[count],&ZOc_Tb3[count],&cp_Tb3[count],
                                        &mwt_Tb3[count],&hiyi_Tb3[count],&Tf_Tb3[count],&nu_Tb3[count],&Ycmax_Tb3[count]
                                     );
                        }

                        if(NY > 0)   
                        {
                        
                         f9=fscanf(
                                     table,fmt_nYis,&Yi01_Tb3[count],&Yi02_Tb3[count],&Yi03_Tb3[count]
                                  );   
                                  
                         //- loop speciesNames
                    /*   forAll(Yi_Tb3,speciesI)  
                         {
                            f9=fscanf(
                                        table,fmt_nYis,&Yi_Tb3[speciesI][count]
                                    );   //再读入'H2O','CO','CO2'的数据
                         }  
                    */
                        }

                        count++;
                    }
                }
            }
        }
        if(ii%10 == 0)  
        {
            prog = 100.0*ii/(NZ-1);
            Info<< " Progress -- " << prog << "%" << endl;
        }
    }

    //- findthe maxmum PV value from Ycmax_Tb3
    const scalar Ycmaxall{*std::max_element(Ycmax_Tb3,Ycmax_Tb3+count+1)};   
    cMaxAll_ = Ycmaxall;

    if(!scaledPV_)
    {
        Info<< "\nunscaled PV -- Ycmaxall = "<<Ycmaxall<< endl;   
    }

    if(flameletT_)  
    {
        delete [] cp_Tb3;
        delete [] hiyi_Tb3;
    }

    Info<< "\nReading non-premixed properties\n" << endl;   
    d2Yeq_Tb2={ new double[NZ*NGZ]{} };
    int countN = 0;
    for(int ii = 0; ii < NZ; ii++)
    {
        for(int jj = 0; jj < NGZ; jj++)
        {
             f10=fscanf( table, "%lf\n", &d2Yeq_Tb2[countN] );   

            countN++;
        }
    }

    fclose(table);

    Info<< "* * * * * * * * * * Complete FlaRe table * * * * * * * * * \n"
        << endl;


/*
   forAll(Yi_Tb3, speciesI)
   {
        Yi_Tb3.set
        (
            speciesI,
            new scalarField(NZ*NC*NGZ*NGC*NZC)
        );
   }
*/ 