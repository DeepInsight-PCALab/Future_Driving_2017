python train.py --dataroot /data3/XJTU2017/task_1/Tusimple/ --name lane_resnext101 --gpu_ids 5 --batchSize 1

python test.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/TE0530/ --results_dir ./tusimple_tests/ --cls_thres 0.9

python test_xml.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml/ --cls_thres 0.9

###################################################################################################################
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_9000 --gpu_ids 0,1 --batchSize 32
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_9000_cpustore --gpu_ids 4,3 --batchSize 32

python test_xml.py --name lane_resnext101_9000_cpustore --gpu_ids -1 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cpu/ --cls_thres 0.5
