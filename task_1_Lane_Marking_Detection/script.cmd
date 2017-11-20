python train.py --dataroot /data3/XJTU2017/task_1/Tusimple/ --name lane_resnext101 --gpu_ids 5 --batchSize 1

python train.py --dataroot /data3/XJTU2017/task_1/Tusimple/ --name lane_resnext101 --gpu_ids 5 --batchSize 1 --model resnext_cls --pretrain ./checkpoints/lane_resnext101_9000_cpustore/latest_net_resnext101.pth --finetune_cls True 

python test.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/TE0530/ --results_dir ./tusimple_tests/ --cls_thres 0.9

python test_xml.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml/ --cls_thres 0.9
python test_xml.py --name lane_resnext101_9000_cpustore --gpu_ids -1 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cpu/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext101_cls --depth 101 --gpu_ids 6 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls/ --cls_thres 0.9

###################################################################################################################
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_9000 --gpu_ids 0,1 --batchSize 32
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_9000_cpustore --gpu_ids 4,3 --batchSize 32
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_cls --gpu_ids 1,0,5 --batchSize 48 --model resnext_cls --pretrain ./checkpoints/lane_resnext101_9000_cpustore/latest_net_resnext101.pth --finetune_cls 0  --schedule 100,150,200,225
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls --gpu_ids 2,3,4 --batchSize 30 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 100,150,200,225
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls_ud3 --gpu_ids 1,0,5,6 --batchSize 48 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 100,150,200,225


python test_xml.py --name lane_resnext101_9000_cpustore --gpu_ids -1 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cpu/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext101_cls --depth 101 --gpu_ids 6 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext101_cls --depth 101 --gpu_ids 6 --test_dir /data3/XJTU2017/tmpdata/test/ --results_xml_dir /data3/XJTU2017/tmpdata/results_test/ --cls_thres 0.9
