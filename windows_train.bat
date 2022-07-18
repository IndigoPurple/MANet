python train_FlowNet_ms_warping_multi.py ^
--batch_size_train 1 ^
--batch_size_test 1 ^
--dataset_use_crop 1 ^
--dataset_crop_H 64 ^
--dataset_crop_W 64 ^
--dataset_workers 1 ^
--test_or_train test ^
--test_snapshot_file 2.pth

REM python train_FlowNet_ms_warping2.py ^
REM --batch_size_train 2 ^
REM --batch_size_test 2 ^
REM --dataset_use_crop 1 ^
REM --dataset_crop_H 64 ^
REM --dataset_crop_W 64 ^
REM --dataset_workers 1