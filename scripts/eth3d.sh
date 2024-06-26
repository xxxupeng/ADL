CUDA_VISIBLE_DEVICES="0" python -W ignore val.py \
    --dataset eth3d \
    --datapath /data/xp/ETH3D \
    --testlist ./datalists/eth3d_all.txt \
    --start_model 0 --end_model 44 --gap 1 \
    --test_batch_size 1 \
    --maxdisp 192 \
    --model PSMNet \
    --estimator DME \
    --model_name  ETH3D_Generalization \
    --loadmodel /data/xp/Check_Point/SceneFlow/SceneFlow_PSMNet_ADL