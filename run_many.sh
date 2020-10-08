#!/bin/bash

VIS_THRESHS=(1.0)
# SUBJS=(ABF10 BB10 GPMF10 GSF10 MC1 MDF10 ND2 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)
# SUBJS=(BB10 GPMF10 MC1 MDF10 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)
SUBJS=(ABF10)

for v in "${VIS_THRESHS[@]}"
do
    for s in "${SUBJS[@]}"
    do
        python3 reconstruct_model_tsdf.py $s --min_ratio_valid $v
    done
done

#python3 reconstruct_model_tsdf.py ABF10 0.7

#python3 viewpoint_selector.py ABF10
#python3 viewpoint_selector.py BB10
#python3 viewpoint_selector.py GPMF10
#python3 viewpoint_selector.py MC1
#python3 viewpoint_selector.py MDF10
#python3 viewpoint_selector.py SB10
#python3 viewpoint_selector.py ShSu10
#python3 viewpoint_selector.py SiBF10
#python3 viewpoint_selector.py SiS1
#python3 viewpoint_selector.py SM2
#python3 viewpoint_selector.py SMu1
#python3 viewpoint_selector.py SS1