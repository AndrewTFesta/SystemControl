basePath = "D:\rit\thesis\SystemControl";
baseDir = fullfile(basePath, "data", "ProjectBCI", "2D");
if ~exist(baseDir, 'dir')
    mkdir(baseDir);
end
addpath(baseDir);
    
load('Subject1_2D.mat')
baseFieldNameList = [
    "Backward1","Backward2","Backward3",...
    "Forward1","Forward2","Forward3",...
    "BackwardImagined","ForwardImagined","Leg",...
];

for fieldNameIdx = 1 : length(baseFieldNameList)
    baseFieldName = baseFieldNameList(fieldNameIdx);
    
    rightName = sprintf("Right%s", baseFieldName);
    rightArr = eval(rightName);
    leftName = sprintf("Left%s", baseFieldName);
    lefttArr = eval(leftName);
    
    fprintf("Converting: %s\n", rightName);
    saveDataFile(rightName, rightArr, baseDir);
    
    fprintf("Converting: %s\n", leftName);
    saveDataFile(leftName, lefttArr, baseDir);
end

function saveDataFile(arrName, arrData, baseDir)
    fname = fullfile(baseDir, sprintf("%s.csv", arrName));
    fileId = fopen(fname, 'w+');

    header = "FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7, F8,T3,T4,T5,T6,FZ,CZ,PZ";
    fprintf(fileId, '%s\n', header);

    sizeArr = size(arrData);
    for rowIdx = 1 : sizeArr(1)
        separator = '';
        for colIdx = 1 : sizeArr(2)
            dataVal = arrData(rowIdx, colIdx);
            fprintf(fileId, '%s%d', separator, dataVal);
            separator = ',';
        end
        fprintf(fileId, '\n');
    end

    fclose(fileId);
end
