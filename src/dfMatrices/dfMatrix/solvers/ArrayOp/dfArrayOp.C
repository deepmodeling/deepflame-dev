#include "dfArrayOp.H"
#include <cmath>

namespace Foam{


scalar dfSumMag(const scalar* __restrict__ const arrPtr, label arrayLen){
    scalar sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (label i = 0; i < arrayLen; i++){
        sum += std::abs(arrPtr[i]);
    }
    return sum;
}

scalar dfSumSqr(const scalar* __restrict__ const arrPtr, label arrayLen){
    scalar sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (label i = 0; i < arrayLen; i++){
        sum += arrPtr[i] * arrPtr[i];
    }
    return sum;
}

scalar dfSumProd(const scalar* __restrict__ const arrAPtr, const scalar* __restrict__ const arrBPtr, label arrayLen){
    scalar sum = 0.;
    #pragma omp parallel for reduction(+:sum)
    for (label i = 0; i < arrayLen; i++){
        sum += arrAPtr[i] * arrBPtr[i];
    }
    return sum;
}

}
