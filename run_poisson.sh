#!/bin/bash

VIS_THRESHS=(0.7 0.6 0.4 0.3 0.2)
SUBJS=(ABF10 BB10 GPMF10 GSF10 MC1 MDF10 ND2 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)
SETTS=_GT_start0_max-1_skip1_segHO3D_renFilter_visRatio

for v in "${VIS_THRESHS[@]}"
do
    v_str=$(echo $v | tr '.' '-')
    for s in "${SUBJS[@]}"
    do
        python3.7 tsdf_to_poisson.py ${s}${SETTS}${v_str}_tsdf.ply
    done
done

# python3.7 tsdf_to_poisson.py ABF_GT_start0_max-1_skip50_segHO3D_renFilter_visRatio0-7_tsdf.ply
