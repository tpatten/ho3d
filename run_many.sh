#!/bin/bash

VIS_THRESHS=(0.7 0.6 0.4 0.3 0.2)
SUBJS=(ABF10 BB10 GPMF10 GSF10 MC1 MDF10 ND2 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)

for v in "${VIS_THRESHS[@]}"
do
    for s in "${SUBJS[@]}"
    do
        python3 reconstruct_model_tsdf.py $s $v
    done
   # do whatever on $i
done

#python3 reconstruct_model_tsdf.py ABF10 0.7
