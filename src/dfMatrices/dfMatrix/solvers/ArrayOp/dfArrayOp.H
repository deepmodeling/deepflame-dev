#pragma once
#include "label.H"
#include "scalar.H"

namespace Foam{

scalar dfSumMag(const scalar* __restrict__ const arrPtr, label arrayLen);

scalar dfSumSqr(const scalar* __restrict__ const arrPtr, label arrayLen);

scalar dfSumProd(const scalar* __restrict__ const arrAPtr, const scalar* __restrict__ const arrBPtr, label arrayLen);

}
