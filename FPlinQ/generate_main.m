clc
clear
clear all

data_size_set = [500,1000];%number of network layouts
link_num_set = [50]; %number of D2D pairs
for i = 1:length(link_num_set)
    num = link_num_set(i);%single-antenna transceivers pairs
    for j = 1:length(data_size_set)
        disp('####### Generate Training Data #######');
        data_size = data_size_set(j);
        [Channel,Label,Distance,Distance_quan]=generate(num,data_size);
        disp('#######Done #######');
        save(sprintf('./mat/dataset_%d_%d.mat',data_size, num),'Channel','Label','Distance','Distance_quan');
    end
end


