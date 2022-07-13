file_list = dir('**/*.mat');

for i=1:length(file_list)
   load([file_list(i).folder, '/', file_list(i).name]) 
   save([file_list(i).folder, '/', file_list(i).name])
end