#!/bin/bash

#python3.7 -m pip install meshlabxml

VIS_THRESHS=(0.3)
# SUBJS=(ABF10 BB10 GPMF10 GSF10 MC1 MDF10 ND2 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)
SUBJS=(ABF10 BB10 GPMF10 MC1 MDF10 SB10 ShSu10 SiBF10 SiS1 SM2 SMu1 SS1)
#SETTS=_GT_start0_max-1_skip1_segHO3D_renFilter_visRatio
SETTS=_views_Uniform_Segmentation_step

for v in "${VIS_THRESHS[@]}"
do
    v_str=$(echo $v | tr '.' '-')
    for s in "${SUBJS[@]}"
    do
        python3.7 tsdf_to_poisson.py ${s}${SETTS}${v_str}_tsdf.ply
        #python3.7 tsdf_to_poisson.py ${s}${SETTS}.off
    done
done

# python3.7 tsdf_to_poisson.py ABF_GT_start0_max-1_skip50_segHO3D_renFilter_visRatio0-7_tsdf.ply
