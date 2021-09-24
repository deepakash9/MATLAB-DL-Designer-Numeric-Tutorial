%% adapted from: https://www.mathworks.com/help/deeplearning/ug/train-network-on-data-set-of-numeric-features.html
% pre-processing as provided in the above link

filename = "transmissionCasingData.csv";
tbl = readtable(filename,'TextType','String');

labelName = "GearToothCondition";
tbl = convertvars(tbl,labelName,'categorical');

categoricalInputNames = ["SensorCondition" "ShaftCondition"];
tbl = convertvars(tbl,categoricalInputNames,'categorical');

for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(tbl(:,name));
    tbl = addvars(tbl,oh,'After',name);
    tbl(:,name) = [];
end
tbl = splitvars(tbl);

numObservations = size(tbl,1)
numObservationsTrain = floor(0.7*numObservations)
numObservationsValidation = floor(0.3*numObservations)
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);

tblTrain = tbl(idxTrain,:);
tblValidation = tbl(idxValidation,:);

%% save as separate csv files
%writetable(tblTrain,'tCasing3Train.csv','Delimiter',',')

%% load as textDatastore

ds3Train = tabularTextDatastore("tCasing3Train.csv");
preview(ds3Train)
ds3newTrain = transform(ds3Train, @(x) [cellfun(@transpose,mat2cell(x{:,1:22},ones(1,149)),'UniformOutput',false) , mat2cell(categorical(x{:,23}),ones(1,149))]);
preview(ds3newTrain)

ds3Valid = tabularTextDatastore("tCasing3Valid.csv");
preview(ds3Valid)
ds3newValid = transform(ds3Valid, @(x) [cellfun(@transpose,mat2cell(x{:,1:22},ones(1,59)),'UniformOutput',false) , mat2cell(categorical(x{:,23}),ones(1,59))]);
preview(ds3newValid)

%% make same network in the Deep Network designer, as given in the link