#include "dfDIASubMatrix.H"
#include <cassert>

namespace Foam{

dfDIASubMatrix::dfDIASubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,scalar>>& rcvList):dfBlockSubMatrix(nRows, nCols){
    // count distance
    std::vector<label> distance_map(2 * row_ - 1, 0);
    for(const auto& entry: rcvList){
        label r = std::get<0>(entry);
        label c = std::get<1>(entry);
        distance_map[r - c + row_ - 1] += 1;
    }
    // distance_count_
    distance_count_ = 0;
    for(label i = 0; i < 2 * row_ - 1; ++i){
        if(distance_map[i] > 0){
            distance_count_ += 1;
        }
    }
    // fill distance_list_
    distance_list_ = std::make_unique<label[]>(distance_count_);
    label idx = 0;
    for(label i = 0; i < 2 * row_ - 1; ++i){
        if(distance_map[i] > 0){
            distance_list_[idx] = i - row_ + 1;
            distance_map[i] = idx;
            idx += 1;
        }else{
            distance_map[i] = -1;
        }
    }

    // fill off_diag_value
    values_ = std::make_unique<scalar[]>(distance_count_ * row_);

    for(label i = 0; i < distance_count_ * row_; ++i){
        values_[i] = 0.0;
    }

    for(const auto& entry: rcvList){
        label r = std::get<0>(entry);
        label c = std::get<1>(entry);
        scalar value = std::get<2>(entry);
        label distance = r - c;
        label dia_col = distance_map[distance + row_ - 1];
        label index = dia_col * row_ + r;
        values_[index] = value;
    }

    value_ldu_idx_ = nullptr;
}

dfDIASubMatrix::dfDIASubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,label>>& rciList):dfBlockSubMatrix(nRows, nCols){
    // count distance
    std::vector<label> distance_map(2 * row_ - 1, 0);
    for(const auto& entry: rciList){
        label r = std::get<0>(entry);
        label c = std::get<1>(entry);
        distance_map[r - c + row_ - 1] += 1;
    }
    // distance_count_
    distance_count_ = 0;
    for(label i = 0; i < 2 * row_ - 1; ++i){
        if(distance_map[i] > 0){
            distance_count_ += 1;
        }
    }
    // fill distance_list_
    distance_list_ = std::make_unique<label[]>(distance_count_);
    label idx = 0;
    for(label i = 0; i < 2 * row_ - 1; ++i){
        if(distance_map[i] > 0){
            distance_list_[idx] = i - row_ + 1;
            distance_map[i] = idx;
            idx += 1;
        }else{
            distance_map[i] = -1;
        }
    }

    // fill off_diag_value
    values_ = std::make_unique<scalar[]>(distance_count_ * row_);
    value_ldu_idx_ = std::make_unique<label[]>(distance_count_ * row_);

    for(label i = 0; i < distance_count_ * row_; ++i){
        values_[i] = 0.0;
        value_ldu_idx_[i] = -1;
    }

    for(const auto& entry: rcvList){
        label r = std::get<0>(entry);
        label c = std::get<1>(entry);
        label faceIndex = std::get<2>(entry);
        label distance = r - c;
        label dia_col = distance_map[distance + row_ - 1];
        label index = dia_col * row_ + r;
        value_ldu_idx_[index] = faceIndex;
    }
}

void dfDIASubMatrix::valueCopyOffDiagBlock(const scalar* const __restrict__ lduValuePt){}

void dfDIASubMatrix::valueCopyDiagBlock(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper){}

void dfDIASubMatrix::SpMV(scalar* const __restrict__ ApsiPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {}

void dfDIASubMatrix::SumA(scalar* const __restrict__ sumAPtr_offset) const {}

void dfDIASubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {}

void dfDIASubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset) const {}

}
