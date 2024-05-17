CUDA_VISIBLE_DEVICES="0,1" python -W ignore train_DDP.py \
    --dataset kitti \
    --datapath /data/xp/KITTI_2015 \
    --trainlist ./datalists/kitti15_train.txt \
    --epochs 600 --lr 0.001  \
    --batch_size 4 \
    --maxdisp 192 \
    --model PSMNet \
    --loss_func ADL \
    --savemodeldir /data/xp/Check_Point/KITTI/   \
    --model_name  KITTI15_PSMNet_ADL \
    --loadmodel /data/xp/Check_Point/SceneFlow/SceneFlow_PSMNet_ADL_train_44.tar

# CUDA_VISIBLE_DEVICES="0" python -W ignore val.py \
#     --dataset kitti \
#     --datapath /data/xp/KITTI_2015 \
#     --testlist ./datalists/kitti15_val.txt \
#     --start_model 19 --end_model 599 --gap 20 \
#     --test_batch_size 1 \
#     --maxdisp 192 \
#     --model PSMNet \
#     --estimator DME \
#     --model_name  KITTI15_PSMNet_ADL \
#     --loadmodel /data/xp/Check_Point/KITTI/KITTI15_PSMNet_ADL

