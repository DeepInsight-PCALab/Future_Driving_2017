python train.py --dataroot /data3/XJTU2017/task_1/Tusimple/ --name lane_resnext101 --gpu_ids 5 --batchSize 1

python train.py --dataroot /data3/XJTU2017/task_1/Tusimple/ --name lane_resnext101 --gpu_ids 5 --batchSize 1 --model resnext_cls --pretrain ./checkpoints/lane_resnext101_9000_cpustore/latest_net_resnext101.pth --finetune_cls True 

python test.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/TE0530/ --results_dir ./tusimple_tests/ --cls_thres 0.9

python test_xml.py --name lane_resnext101 --gpu_ids 0 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml/ --cls_thres 0.9
python test_xml.py --name lane_resnext101_9000_cpustore --gpu_ids -1 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cpu/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext101_cls --depth 101 --gpu_ids 6 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls/ --cls_thres 0.9

###################################################################################################################
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_debug --gpu_ids 7 --batchSize 12 --model resnext_cls --depth 101 --finetune_cls 0 --cutout 20 --schedule 100,150,200,225
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_9000 --gpu_ids 0,1 --batchSize 32
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_9000_cpustore --gpu_ids 4,3 --batchSize 32
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext101_cls --gpu_ids 1,0,5 --batchSize 48 --model resnext_cls --pretrain ./checkpoints/lane_resnext101_9000_cpustore/latest_net_resnext101.pth --finetune_cls 0  --schedule 100,150,200,225
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls --gpu_ids 2,3,4 --batchSize 30 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 100,150,200,225

python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls_largeup --gpu_ids 2,3,4 --batchSize 30 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 100,150,200,225 --cutout 40

python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls_ud3 --gpu_ids 1,0,5,6 --batchSize 48 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 100,150,200,225
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_320x256 --gpu_ids 1,0,5,6 --batchSize 48 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 100,150,200,225 --cutout 40
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_320x256_finetune --pretrain checkpoints/lane_resnext152_320x256/latest_net_resnext152.pth --json xjtrain --gpu_ids 1,0,5,6 --batchSize 48 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 50,75,100,125 --cutout 40
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_320x256_finetune --pretrain checkpoints/lane_resnext152_320x256/latest_net_resnext152.pth --json xjtrain --gpu_ids 1,0 --batchSize 24 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 50,75,100,125 --cutout 40
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_320x256_finetune_down --pretrain checkpoints/lane_resnext152_320x256/latest_net_resnext152.pth --json xjtrain --gpu_ids 5,6 --batchSize 24 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 50,75,100,125 --cutout 40
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls_largeup_ft_all --pretrain checkpoints/lane_resnext152_cls_largeup/latest_net_resnext152.pth --json train --gpu_ids 4,5,6,7 --batchSize 48 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 50,75,100,125 --cutout 40
python train.py --dataroot /data3/DeepInsight/Lane_PAMI/Datasets/tolabel/ --name lane_resnext152_cls_largeup_ft_xj --pretrain checkpoints/lane_resnext152_cls_largeup/latest_net_resnext152.pth --json xjtrain --gpu_ids 0,1,2,3 --batchSize 48 --model resnext_cls --depth 152 --finetune_cls 0  --schedule 50,75,100,125 --cutout 40


###################################################################################################################
python test_xml.py --name lane_resnext101_9000_cpustore --gpu_ids -1 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cpu/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext101_cls --depth 101 --gpu_ids 6 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext101_cls --depth 101 --gpu_ids 6 --test_dir /data3/XJTU2017/tmpdata/test/ --results_xml_dir /data3/XJTU2017/tmpdata/results_test/ --cls_thres 0.9
python test_cls_xml.py --name lane_resnext152_320x256 --depth 152 --gpu_ids 7 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls_320x256_gbk/ --cls_thres 0.9 --nms_thres 20
python test_cls_xml.py --name lane_resnext152_cls_largeup --depth 152 --gpu_ids 7 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls_largeup/ --cls_thres 0.9 --nms_thres 20
python test_cls_xml.py --name lane_resnext152_cls_largeup --depth 152 --gpu_ids 7 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls_largeup_00/ --cls_thres 0.9 --nms_thres 30
python test_cls_xml.py --name lane_resnext152_cls_largeup --depth 152 --gpu_ids 2 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls_largeup_01/ --cls_thres 0.9 --nms_thres 30
python test_cls_xml.py --name lane_resnext152_cls_largeup --depth 152 --gpu_ids 2 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_cls_largeup_02/ --cls_thres 0.9 --nms_thres 30
python test_cls_xml.py --name lane_resnext152_320x256_finetune --depth 152 --gpu_ids 0 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_320x256_finetune/ --cls_thres 0.9 --nms_thres 30
python test_cls_xml.py --name lane_resnext152_320x256_finetune_down --depth 152 --gpu_ids 1 --test_dir /data3/XJTU2017/task_1/TSD-Lane/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_320x256_finetune_down/ --cls_thres 0.9 --nms_thres 30
python test_cls_xml.py --name lane_resnext152_320x256_finetune --depth 152 --gpu_ids 1 --test_dir /data3/XJTU2017/task_1/Tusimple_test/ --results_xml_dir /data3/XJTU2017/tmpdata/results_xml_320x256_finetune_tusimpletest/ --cls_thres 0.9 --nms_thres 30



###################################################################################################################
python test_cls_xml.py --name task_1 --depth 152 --gpu_ids 0 --test_dir /home/liangchen/Desktop/FC2017/task_1/TSD-Lane/ --results_xml_dir /home/liangchen/Desktop/Results/task_1/ --cls_thres 0.9
