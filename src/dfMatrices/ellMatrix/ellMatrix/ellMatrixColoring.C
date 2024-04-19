/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFoam: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFoam Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFoam.

    OpenFoam is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFoam is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFoam.  If not, see <http://www.gnu.org/licenses/>.

Description
    Multiply a given vector (second argument) by the matrix or its transpose
    and return the result in the first argument.

\*---------------------------------------------------------------------------*/

#include "ellMatrix.H"
#include <cassert>
#include <omp.h>

// #define HASH_BETA 997
#define HASH_BETA 997

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// void Foam::ellMatrix::build_graph() {
//     blocked_off_diag_colidx_.resize(block_count_ * max_count_ * row_block_size_);
//     blocked_off_diag_count_.resize(block_count_);

//     // build graph of blocked matrix
//     #pragma omp parallel
//     {
//         // #pragma omp for
//         // for(size_t i = 0; i < blocked_off_diag_colidx_.size(); ++i){
//         //     blocked_off_diag_colidx_[i] = -1;
//         // }
//         #pragma omp for
//         for(label i = 0; i < blocked_off_diag_count_.size(); ++i){
//             blocked_off_diag_count_[i] = 0;
//         }    

//         labelList tmp(row_block_size_ * max_count_);
//         label tmp_size = 0;
        
//         #pragma omp for
//         for(label bi = 0; bi < block_count_; ++bi){
//             label rbs = ELL_BLOCK_START(bi);
//             label rbe = ELL_BLOCK_END(rbs);
//             label rbl = ELL_BLOCK_LEN(rbs,rbe);
//             label index_block_start = ELL_INDEX_BLOCK_START(rbs);
//             label* blocked_off_diag_colidx_current = blocked_off_diag_colidx_.begin() + bi * row_block_size_ * max_count_;
//             // expand
//             for(label ellcol = 0; ellcol < max_count_; ++ellcol){
//                 label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
//                 for(label br = 0; br < rbl; ++br){
//                     label row = rbs + br;
//                     label index = index_ellcol_start + br;
//                     label col = off_diag_colidx_[index];
//                     if(row == col) continue; // 跳过填充的非零元
//                     label block_index_col = ELL_BLOCK_INDEX(col);
//                     if(bi == block_index_col) continue; // 忽略块内依赖
//                     label blocked_index = blocked_off_diag_count_[bi];
//                     blocked_off_diag_colidx_current[blocked_index] = block_index_col;
//                     blocked_off_diag_count_[bi] += 1;
//                 }
//             }
//             label blocked_off_diag_count_current = blocked_off_diag_count_[bi];
//             // sort
//             std::sort(blocked_off_diag_colidx_current, blocked_off_diag_colidx_current + blocked_off_diag_count_current);
//             // union
//             if(blocked_off_diag_count_current > 0){
//                 tmp[0] = blocked_off_diag_colidx_current[0];
//                 tmp_size = 1;
//                 label prev = blocked_off_diag_colidx_current[0];
//                 for(label i = 1; i < blocked_off_diag_count_current; ++i){
//                     if(blocked_off_diag_colidx_current[i] != prev){
//                         prev = blocked_off_diag_colidx_current[i];
//                         tmp[tmp_size] = blocked_off_diag_colidx_current[i];
//                         tmp_size += 1;
//                     }
//                 }
//                 for(label i = 0; i < tmp_size; ++i){
//                     blocked_off_diag_colidx_current[i] = tmp[i];
//                 }
//                 blocked_off_diag_count_[bi] = tmp_size;
//             }

//         }
//     }
// }

// void Foam::ellMatrix::hash_coloring(){
//     // hash coloring
//     labelList old_color(block_count_);
//     labelList new_color(block_count_);
//     for(label bi = 0; bi < block_count_; ++bi){
//         old_color[bi] = -1;
//         new_color[bi] = -1;
//     }

//     num_color_ = 0;
//     label num_colored_node = 0;
//     label colorIter = 0;

//     label hash_beta = HASH_BETA;

//     while(num_colored_node != block_count_){

//         labelList hash_value(block_count_);

//         labelList num_colored_node_thread(omp_get_max_threads());

//         #pragma omp parallel
//         {
//             label num_colored_node_this_thread = 0;

//             #pragma omp for
//             for(label bi = 0; bi < block_count_; ++bi){
//                 hash_value[bi] = bi % hash_beta;
//             } 
//             #pragma omp for
//             for(label bi = 0; bi < block_count_; ++bi){
//                 // 如果该节点还没染色
//                 if(old_color[bi] == -1){
//                     label cur_hash_value = hash_value[bi];

//                     label max = 0;
//                     label min = 0;

//                     // 遍历所有邻居
//                     label* blocked_off_diag_colidx_current = blocked_off_diag_colidx_.begin() + bi * row_block_size_ * max_count_;
//                     label blocked_off_diag_count_current = blocked_off_diag_count_[bi];
//                     for(label i = 0; i < blocked_off_diag_count_current; ++i){
//                         label nei = blocked_off_diag_colidx_current[i];
//                         // 只检查还没有染色的邻居比
//                         if(old_color[nei] == -1){
//                             label nei_hash_value = hash_value[bi];
//                             //比较当前点的hash值和当前邻居的hash值
//                             //如果大于邻居节点则置max为1
//                             if(cur_hash_value > nei_hash_value) max = 1;
//                             //如果小于邻居节点则置min为1
//                             if(cur_hash_value < nei_hash_value) min = 1;
//                             //如果等于邻居节点，则比较行号
//                             //如果行号大于邻居节点则置max为1
//                             //如果行号小于邻居节点则置min为1
//                             if(cur_hash_value == nei_hash_value){
//                                 if(bi > nei){
//                                     max = 1;
//                                 }
//                                 else{
//                                     min = 1;
//                                 }
//                             }
//                         }
//                     }

//                     //如果当前节点是局部最大则将其染色为num_color_，如果当前节点是局部最小则将其染色为num_color_ + 1
//                     //如果min和max同为1，则不染色

//                     //如果同为0，说明该点所有邻居都被染过了，按局部最小染色 
//                     if(min == 0 && max == 0){
//                         new_color[bi] = num_color_ + 1;
//                         num_colored_node_this_thread += 1;
//                     }
//                     //只有min为1则按局部最小染色，只有max为1则按局部最大染色
//                     if(max == 1 && min == 0){
//                         new_color[bi] = num_color_;
//                         num_colored_node_this_thread += 1;
//                     }
//                     if(min == 1 && max == 0){                
//                         new_color[bi] = num_color_ + 1;
//                         num_colored_node_this_thread += 1;
//                     }
//                 }
//             }
        
//             label tid = omp_get_thread_num();
//             num_colored_node_thread[tid] = num_colored_node_this_thread;
//         }

        
//         for(label i = 0; i < num_colored_node_thread.size(); ++i){
//             num_colored_node += num_colored_node_thread[i];
//         }

//         // update old color
//         for(label bi = 0; bi < block_count_; ++bi){
//             old_color[bi] = new_color[bi];
//         }

//         num_color_ += 2;

//         // Info << "HashColor Iter : " << colorIter << endl;
//         // Info << "\tNumber of color : " << num_color_ << endl;
//         // Info << "\tNumber of colored node : " << num_colored_node << endl;
//         // Info << "\tNumber of total node : " << block_count_ << endl;
        
//         colorIter += 1;
//     }
//     // update color
//     color_of_node_.resize(block_count_);
//     for(label bi = 0; bi < block_count_; ++bi){
//         color_of_node_[bi] = old_color[bi];
//     }
// }

// void Foam::ellMatrix::jpl_coloring(){
//     // hash coloring
//     labelList old_color(block_count_);
//     labelList new_color(block_count_);
//     for(label bi = 0; bi < block_count_; ++bi){
//         old_color[bi] = -1;
//         new_color[bi] = -1;
//     }

//     num_color_ = 0;
//     label num_colored_node = 0;
//     label colorIter = 0;

//     label hash_beta = HASH_BETA;

//     while(num_colored_node != block_count_){

//         labelList hash_value(block_count_);

//         labelList num_colored_node_thread(omp_get_max_threads());

//         #pragma omp parallel
//         {
//             label num_colored_node_this_thread = 0;

//             #pragma omp for
//             for(label bi = 0; bi < block_count_; ++bi){
//                 hash_value[bi] = bi % hash_beta;
//             } 
//             #pragma omp for
//             for(label bi = 0; bi < block_count_; ++bi){
//                 // 如果该节点还没染色
//                 if(old_color[bi] == -1){
//                     label cur_hash_value = hash_value[bi];

//                     label max = 1;

//                     // 遍历所有邻居
//                     label* blocked_off_diag_colidx_current = blocked_off_diag_colidx_.begin() + bi * row_block_size_ * max_count_;
//                     label blocked_off_diag_count_current = blocked_off_diag_count_[bi];
//                     for(label i = 0; i < blocked_off_diag_count_current; ++i){
//                         label nei = blocked_off_diag_colidx_current[i];
//                         // 只检查还没有染色的邻居比
//                         if(old_color[nei] == -1){
//                             label nei_hash_value = hash_value[bi];
//                             //比较当前点的hash值和当前邻居的hash值
//                             //如果小于邻居节点则置max为0
//                             if(cur_hash_value < nei_hash_value) max = 0;
//                             //如果等于邻居节点，则比较行号
//                             //如果行号小于邻居节点则置max为0
//                             if(cur_hash_value == nei_hash_value){
//                                 if(bi < nei)
//                                     max = 0;
//                             }
//                         }
//                     }

//                     //如果当前节点是局部最大则将其染色为color_num + 1
//                     //如果染色，给计数器加一
//                     if(max == 1){
//                         new_color[bi] = num_color_;
//                         num_colored_node_this_thread += 1;
//                     }
//                 }
//             }
        
//             label tid = omp_get_thread_num();
//             num_colored_node_thread[tid] = num_colored_node_this_thread;
//         }
        
//         for(label i = 0; i < num_colored_node_thread.size(); ++i){
//             num_colored_node += num_colored_node_thread[i];
//         }

//         // update old color
//         for(label bi = 0; bi < block_count_; ++bi){
//             old_color[bi] = new_color[bi];
//         }

//         num_color_ += 1;

//         // Info << "HashColor Iter : " << colorIter << endl;
//         // Info << "\tNumber of color : " << num_color_ << endl;
//         // Info << "\tNumber of colored node : " << num_colored_node << endl;
//         // Info << "\tNumber of total node : " << block_count_ << endl;
        
//         colorIter += 1;
//     }
//     // update color
//     color_of_node_.resize(block_count_);
//     for(label bi = 0; bi < block_count_; ++bi){
//         color_of_node_[bi] = old_color[bi];
//     }
// }

// void Foam::ellMatrix::color_count(){
//     // count node pre color:
//     labelList node_count_pre_color(num_color_, 0);
//     for(label bi = 0; bi < block_count_; ++bi){
//         label color = color_of_node_[bi];
//         node_count_pre_color[color] += 1;
//     }
//     // compute color_ptr_
//     color_ptr_.resize(num_color_ + 1);
//     labelList color_index(num_color_ + 1);
//     color_ptr_[0] = 0;
//     color_index[0] = 0;
//     for(label c = 0; c < num_color_; ++c){
//         color_ptr_[c + 1] = color_ptr_[c] + node_count_pre_color[c];
//         color_index[c + 1] = color_index[c] + node_count_pre_color[c];
//     }
//     // fill-in nodes_of_color_
//     nodes_of_color_.resize(block_count_);
//     for(label bi = 0; bi < block_count_; ++bi){
//         label color = color_of_node_[bi];
//         label index = color_index[color];
//         nodes_of_color_[index] = bi;
//         color_index[color] += 1;
//     }

//     colored_ = true;
// }

// void Foam::ellMatrix::coloring(){
//     if(colored_)
//         return;

//     build_graph();

//     hash_coloring();
//     // jpl_coloring();
//     color_count();
//     // coloring_analyze();
// }


// void Foam::ellMatrix::coloring_analyze() const {
//     label edge_count = 0;
//     std::vector<label> degree_frequency(max_count_ * row_block_size_ + 1,0);

//     for(label bi = 0; bi < block_count_; ++bi){
//         label num_nei = blocked_off_diag_count_[bi];
//         degree_frequency[num_nei] += 1;
//         edge_count += num_nei;
//     }

//     Info << "ell sparse matrix coloring analyze : --------------------------------------------------------------------" << endl;
//     Info << "number of node : " << block_count_ << endl;
//     Info << "number of edge : " << edge_count << endl;
//     Info << "number of color : " << num_color_ << endl;
//     Info << "degree frequency of subgraph : " << endl;
//     for(size_t i = 0; i < degree_frequency.size(); ++i){
//         if(degree_frequency[i] > 0){
//             Info << "\t" << i << " : " << degree_frequency[i] << endl;
//         }
//     }
//     Info << "number of node per color : " << endl;
//     for(label c = 0; c < num_color_; ++c){
//         Info << "\t" << c << " : " << color_ptr_[c + 1] - color_ptr_[c] << endl;
//     }
//     Info << "------------------------------------------------------------------------------------------------" << endl;
// }
