%% function                                        
function [] = corruption_process(list_file, dataset_path, ...
                                                dataset_path_noise,...
                                                dataset_path_blur,...
                                                dataset_path_downsample)
    fid = fopen(list_file);
    folder_set = textscan(fid,'%s'); folder_set=folder_set{1,1}; N = size(folder_set,1);
    fclose(fid);

    for s=1:N
        tic;
       folder = folder_set{s,1};
       fprintf('%d out of %d', s, N);
       add_noise_to_input(fullfile(dataset_path, folder), ...
                            fullfile(dataset_path_noise, folder))
       blur_input(fullfile(dataset_path, folder), ...
                            fullfile(dataset_path_blur, folder))
       downsample_input(fullfile(dataset_path, folder), ...
                            fullfile(dataset_path_downsample, folder))
        toc;
    end
end