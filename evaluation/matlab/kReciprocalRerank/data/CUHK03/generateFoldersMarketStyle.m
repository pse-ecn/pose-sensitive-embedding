clear;

outputPath = '/tmp/cuhk03';
detected_or_labeled = 0;


if detected_or_labeled == 1
    img_vector = load('cuhk-03.mat', 'detected');
    img_vector = img_vector.detected;
    load('cuhk03_new_protocol_config_detected.mat');
else
    img_vector = load('cuhk-03.mat', 'labeled');
    img_vector = img_vector.labeled;
    load('cuhk03_new_protocol_config_labeled.mat');
end

load('D:\development\private\masters\evaluation\market1501\kReciprocalRerank\data\CUHK03\cuhk03_new_protocol_config_labeled.mat')



if exist(outputPath, 'dir')
    rmdir(outputPath, 's');
end
mkdir(outputPath);


write_folder(outputPath, img_vector, filelist(train_idx), 'train');
write_folder(outputPath, img_vector, filelist(gallery_idx), 'test');
write_folder(outputPath, img_vector, filelist(query_idx), 'query');


function write_folder(outputPath, img_vector, filelist, folder_name)
    path = [outputPath '/' folder_name];
    mkdir(path);

    for img_name_cell = filelist'
        img_name = img_name_cell{1};
        j1 = str2num(img_name(1));
        j2 = str2num(img_name(3:5));
        j3 = str2num(img_name(9:10));
        imwrite(img_vector{j1}{j2,j3}, [path '/' img_name]);  
    end
end