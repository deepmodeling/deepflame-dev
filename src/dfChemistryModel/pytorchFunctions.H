#include <mpi.h>
#include <iostream>
#include "dfChemistryModel.H"
template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN(const DeltaTType &deltaT)
{
    scalar deltaTMin = great;
    // set the cores slaved by a DCU
    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    Info << "=== begin solve_DNN === " << endl;
    if (gpu_)
    {
        Info << "now DNN inference is conducted on GPU" << endl;
    }
    else
    {
        Info << "now DNN inference is conducted on CPU" << endl;
    }

    std::chrono::steady_clock::time_point start0 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();

    std::vector<double> inputsDNN;
    label output_size = mixture_.nSpecies();
    Info << "input_size: "<< output_size + 3 << endl;
    Info << "output_size: "<< output_size << endl;
    
    //frozen //pre
    int celli_num1=0;
    forAll(T_,celli)
    {   
        selectDNN_[celli] = 1;

        if (T_[celli] < Tem)
        {
            Qdot_[celli] = 0;
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                RR_[i][celli] = 0.0;
            }
            selectDNN_[celli] = 0;
        }
        else
        {
            celli_num1+=1;
            inputsDNN.push_back(T_.primitiveField()[celli]); // temperature
            inputsDNN.push_back(p_.primitiveField()[celli]); // pressure
            for(int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                inputsDNN.push_back(Y_[speciID].primitiveField()[celli]);  // mass fraction Y
            }
            inputsDNN.push_back(rho_.primitiveField()[celli]); // rho
        }       
    }

    std::chrono::steady_clock::time_point stop1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime1 = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
    time_getDNNinputs_ += processingTime1.count();

    //python

    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();

    double deltaTValue = deltaT[0];
    Info << "daltaTValue: "<< deltaTValue << endl;
    Info << "DNN_size " << inputsDNN.size()/(3 + mixture_.nSpecies()) << endl;

    int celli_num2 = 0;
    if (celli_num1 > 0) 
    {
        if(batch_on)
        {
            int onebatchnum = batch_size * (3 + mixture_.nSpecies());
            int num_batches = inputsDNN.size() / onebatchnum + 1; // batch numbers
            std::vector<double> all_results;
            int RealInputNum = 0;
            int RealOutputNum = 0;
            for (int batch_index = 0; batch_index < num_batches; ++batch_index)
            {
                // initial number of one batch
                int start_index = batch_index * onebatchnum;
                
                if (batch_index == num_batches - 1)
                {
                    RealInputNum = inputsDNN.size() - start_index;
                }
                else
                {
                    RealInputNum = onebatchnum;
                }
                
                pybind11::array_t<double> input = pybind11::array_t<double>(
                    {RealInputNum}, // 
                    {sizeof(double)}, // 
                    &inputsDNN[start_index] // 
                );
                
                std::chrono::steady_clock::time_point start6 = std::chrono::steady_clock::now();

                pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> result = call_DNN.attr("inference")(input, deltaTValue, Pstream::myProcNo());
                
                std::chrono::steady_clock::time_point stop6 = std::chrono::steady_clock::now();
                std::chrono::duration<double> processingTime6 = std::chrono::duration_cast<std::chrono::duration<double>>(stop6 - start6);
                time_real_pythonTime_ += processingTime6.count();
                
                pybind11::array_t<double> result_array(result);
                double* data_ptr = result_array.mutable_data();

                RealOutputNum = RealInputNum / (3+mixture_.nSpecies())*mixture_.nSpecies();
                all_results.insert(all_results.end(), data_ptr, data_ptr + RealOutputNum);
                data_ptr = nullptr;
            }

            std::chrono::steady_clock::time_point start4 = std::chrono::steady_clock::now();
            forAll(T_, celli)   /// 
            {
                if (selectDNN_[celli] == 1) {
                    for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                    {
                        RR_[speciID][celli] = all_results[output_size * celli_num2 + speciID];
                    }
                    celli_num2 += 1;
                }
            }
            std::chrono::steady_clock::time_point stop4 = std::chrono::steady_clock::now();
            std::chrono::duration<double> processingTime4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop4 - start4);
            time_updateRR_ += processingTime4.count();
        }
        else
        {
            pybind11::array_t<double> input = pybind11::array_t<double>({inputsDNN.size()}, {8}, &inputsDNN[0]);  // inputsDNN.size()  
            std::chrono::steady_clock::time_point start6 = std::chrono::steady_clock::now();
            pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> result = call_DNN.attr("inference")(input, deltaTValue, Pstream::myProcNo());

            std::chrono::steady_clock::time_point stop6 = std::chrono::steady_clock::now();
            std::chrono::duration<double> processingTime6 = std::chrono::duration_cast<std::chrono::duration<double>>(stop6 - start6);
            time_real_pythonTime_ += processingTime6.count();

            pybind11::array_t<double> result_array(result);
            double* all_results = result_array.mutable_data();

            std::chrono::steady_clock::time_point start4 = std::chrono::steady_clock::now();
            forAll(T_, celli)   /// 
            {
                if (selectDNN_[celli] == 1) {
                    for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                    {
                        RR_[speciID][celli] = all_results[output_size * celli_num2 + speciID];
                    }
                    celli_num2 += 1;
                }
            }
            std::chrono::steady_clock::time_point stop4 = std::chrono::steady_clock::now();
            std::chrono::duration<double> processingTime4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop4 - start4);
            time_updateRR_ += processingTime4.count();
        }
    }

    else
    {
        std::cout << "no point will be inferenced in Processor " << Pstream::myProcNo() << nl << endl;
    }

    if (celli_num1 != celli_num2)
    {
        std::cout << "Warning" << nl <<endl;
    }
    
    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
    time_DNNinference_ += processingTime2.count();
    

    //post

    std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();

    int sum = 0;
    if (celli_num1 > 0)
    {    
        forAll(T_, celli)  
        {
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                Qdot_[celli] -= hc_[speciID] * RR_[speciID][celli];
            }
            if (Qdot_[celli] < Qdot_limit)
            {
                sum = sum + 1;
                Qdot_[celli] = 0;
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    RR_[speciID][celli] = 0;
                }
            } 
        }
    }
    Info << "now " << sum << " DNN Qdot is set to zero" << endl; 

    std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
    time_updateQdot_ += processingTime3.count();

    std::chrono::steady_clock::time_point stop0 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime0 = std::chrono::duration_cast<std::chrono::duration<double>>(stop0 - start0);
    time_allsolve_ += processingTime0.count();

    Info << "=== end solve_DNN === " << endl;
    return deltaTMin;
}
