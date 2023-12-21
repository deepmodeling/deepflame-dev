            value_temp = std::vector<std::vector<double>> ((NY+NS),std::vector<double>(singleTableSize_));
            count = 0;    //counter
            for (int hh=0; hh<NH; ++hh)
            {
                for(int ii = 0; ii < NZ; ++ii)
                {
                    for(int jj = 0; jj < NC; ++jj)
                    {
                        for(int kk = 0; kk < NGZ; ++kk)
                        {
                            for(int ll = 0; ll < NGC; ++ll)
                            {
                                for(int mm = 0; mm < NZC; ++mm)
                                {
                                    if (std::getline(table, line))
                                    {
                                        std::istringstream iss(line);
                                        for (int ivar=0; ivar<NY+NS; ++ivar)
                                        {
                                            iss >> value_temp[ivar][count];
                                        } 
                                    }
                                    else
                                    {
                                        Info << "Error: no more line in: h=" << hh << ", Z=" << ii << ", c=" << jj << ", gz=" << kk << ", gc=" << ll <<", zc=" << mm <<endl;
                                    }
                                    count++;
                                }
                            }
                        }
                    }

                    if(ii%10 == 0 && NH == 1) 
                    {
                        prog = 100.0*ii / ((NZ-1));
                        Info << " Progress -- " << prog << "%" << endl;
                    }
                }

                if(NH != 1) 
                {
                    prog = 100.0*hh / ((NH-1));
                    Info << " Progress -- " << prog << "%" << endl;
                }
            }

    
