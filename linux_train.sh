### training 
# python train_FlowNet_ms_warping2.py  --batch_size_train 1 --batch_size_test 1  --test_or_train train --cuda_devices 3
# python train_FlowNet_ms_warping_multi_simple.py  --batch_size_train 1 --batch_size_test 1  --test_or_train train --cuda_devices 0
# python train_FlowNet_ms_warping_multi_simple_efficient.py  --batch_size_train 1 --batch_size_test 1  --test_or_train train --cuda_devices 0    # seems slower than original, obsolete
# python train_FlowNet_ms_warping_multi.py  --batch_size_train 1 --batch_size_test 1  --test_or_train train --cuda_devices 2

### testing
# python train_FlowNet_ms_warping2.py  --batch_size_train 1 --batch_size_test 1  --test_or_train test --test_snapshot_file 11.pth --cuda_devices 0
# python train_FlowNet_ms_warping_multi_simple.py --batch_size_train 1 --batch_size_test 1  --test_or_train test --test_snapshot_file 11.pth --cuda_devices 0 
python train_FlowNet_ms_warping_multi.py  --batch_size_train 1 --batch_size_test 1  --test_or_train test --test_snapshot_file 11.pth --cuda_devices 0
