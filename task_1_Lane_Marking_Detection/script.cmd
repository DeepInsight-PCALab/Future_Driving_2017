python train.py --dataroot /data3/XJTU2017/task_1/Tusimple/ --name lane_resnext101 --gpu_ids 5 --batchSize 1

python test.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/TE0530/ --results_dir ./tusimple_tests/ --cls_thres 0.9
