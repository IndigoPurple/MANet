addpath generate_testing_sample;
dataset_root = '../dataset/vimeo_septuplet';
dataset_path = fullfile(dataset_root, 'sequences');
dataset_path_noise = fullfile(dataset_root, 'sequences_noise');
dataset_path_blur = fullfile(dataset_root, 'sequences_blur');
dataset_path_downsample = fullfile(dataset_root, 'sequences_downsample');
test_list = fullfile(dataset_root, 'sep_testlist.txt');
train_list = fullfile(dataset_root, 'sep_trainlist.txt');

%% train set
corruption_process(train_list, dataset_path, ...
                            dataset_path_noise,...
                            dataset_path_blur,...
                            dataset_path_downsample)
%% test set
corruption_process(test_list, dataset_path, ...
                            dataset_path_noise,...
                            dataset_path_blur,...
                            dataset_path_downsample)




