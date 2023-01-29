#!/bin/sh
# for file in $PWD/* $PWD/***/**/*; do
#     f=$file
#     if [ "${f##*.}"x = "yaml"x ]||[ "${f##*.}"x = "xml"x ]; then
#         echo "mechanism file exist."
#     else    
#         sed -i 's///g' "$file"
#     fi   
# done
for file in $PWD/processor*/**/*; do  
    sed -i 's/,//g' "$file"
done
