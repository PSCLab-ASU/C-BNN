%% Train set & val set prep for Cifar10
clc;
clear;


%% load training data (include val)
fprintf('loading train/valid dataset...\n');
TrainImage = zeros(50000,3072);
TrainLabel = zeros(50000,1);
for i = 1:5
    i_str = num2str(i);
    load (strcat('data_batch_',i_str,'.mat'));
    TrainImage(10000*i-9999:10000*i,:) = data;
    TrainLabel(10000*i-9999:10000*i,:) = labels;
end
TrainImage = reshape(TrainImage', [32, 32, 3, 50000]);
TrainImage = permute(TrainImage, [2 1 3 4]); % image flip
TrainImage = permute(TrainImage, [4 3 1 2]); % format in 50000x3x32x32
TrainLabel = TrainLabel + 1; % 0-9 to 1-10


%% convert training dataset of cifar10
fprintf('converting train/valid dataset...\n');
TrainImageBi = zeros(50000,3*8,32,32);

for i=1:50000
    for j=1:3
        TrainImageTp = TrainImage(i,j,:,:);
        TrainImageRs = reshape(TrainImageTp,[32*32,1]);
        TrainImageRsBi = de2bi(TrainImageRs,8);
        cnt = 1;
        for k=1:32
            for q=1:32
                TrainImageBi(i,j*8-7:j*8,k,q) = TrainImageRsBi(cnt,:);
                cnt = cnt + 1;
            end
        end
    end
end

%% load test dataset of cifar10, and save
fprintf('loading test dataset...\n');
load test_batch.mat
TestImage = reshape(data', [32, 32, 3, 10000]);
TestImage = permute(TestImage, [2 1 3 4]); % image flip
TestImage = permute(TestImage, [4 3 1 2]); % format in 10000x3x32x32
TestLabel = labels + 1; % 0-9 to 1-10


%% convert test dataset of cifar10
fprintf('converting test dataset...\n');
TestImageBi = zeros(10000,3*8,32,32);

for i=1:10000
    for j=1:3
        TestImageTp = TestImage(i,j,:,:);
        TestImageRs = reshape(TestImageTp,[32*32,1]);
        TestImageRsBi = de2bi(TestImageRs,8);
        cnt = 1;
        for k=1:32
            for q=1:32
                TestImageBi(i,j*8-7:j*8,k,q) = TestImageRsBi(cnt,:);
                cnt = cnt + 1;
            end
        end
    end
end

%% Partition data
train_set_x = TrainImageBi(1:45000,:,:,:);
train_set_y = TrainLabel(1:45000,:);
valid_set_x = TrainImageBi(45001:50000,:,:,:);
valid_set_y = TrainLabel(45001:50000,:);
test_set_x = TestImageBi;
test_set_y = TestLabel;

%% Data argumentation - flipping
%% flip data
train_set_x_flip = flip(train_set_x,3);
valid_set_x_flip = flip(valid_set_x,3);

%% append data
train_set_x = cat(1,train_set_x,train_set_x_flip);
train_set_y = cat(1,train_set_y,train_set_y);
valid_set_x = cat(1,valid_set_x,valid_set_x_flip);
valid_set_y = cat(1,valid_set_y,valid_set_y);

%% Export data
fprintf('export data...\n');
mkdir data_gen;
save('data_gen/train_set_x.mat','train_set_x','-v7.3');
fprintf('train_set_x done\n');
save('data_gen/train_set_y.mat','train_set_y','-v7.3');
fprintf('train_set_y done\n');
save('data_gen/valid_set_x.mat','valid_set_x','-v7.3');
fprintf('valid_set_x done\n');
save('data_gen/valid_set_y.mat','valid_set_y','-v7.3');
fprintf('valid_set_y done\n');
save('data_gen/test_set_x.mat','test_set_x','-v7.3');
fprintf('test_set_x done\n');
save('data_gen/test_set_y.mat','test_set_y','-v7.3');
fprintf('test_set_y done\n');

