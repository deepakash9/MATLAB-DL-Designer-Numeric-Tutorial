%% adapted from: https://www.mathworks.com/help/deeplearning/ug/train-network-on-data-set-of-numeric-features.html
% pre-processing as provided in the above link
data = chickenpox_dataset;
data = [data{:}];

numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

writematrix(dataTrain,'TSTrain.csv','Delimiter',',');
writematrix(dataTest,'TSTest.csv','Delimiter',',');

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
allTrain = [XTrain', YTrain'];

%% save as separate csv files
writematrix(allTrain,'TSTrainStd.csv','Delimiter',',');

%% load as textDatastore
dsTrain = tabularTextDatastore("TSTrainStd.csv");
preview(dsTrain)
dsnewTrain = transform(dsTrain, @(x) [mat2cell(x{:,1},ones(1,448)), mat2cell(x{:,2},ones(1,448))]);
preview(dsnewTrain)

%% make same network in the Deep Network designer, as given in the link