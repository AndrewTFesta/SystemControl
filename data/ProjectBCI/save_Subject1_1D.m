basePath = "D:\rit\thesis\SystemControl";
baseDir = fullfile(basePath, "data", "ProjectBCI", "1D");
if ~exist(baseDir, 'dir')
    mkdir(baseDir);
end
addpath(baseDir);
    
load('Subject1_1D.mat')
filenameList = [
    "baseline", "left", "right"
];

for filenameIdx = 1 : length(filenameList)
    dataFname = filenameList(filenameIdx);
    dataArr = eval(dataFname);
    
    fprintf("Converting: %s\n", dataFname);
    saveDataFile(dataFname, dataArr, baseDir);
end

function saveDataFile(arrName, arrData, baseDir)
    fname = fullfile(baseDir, sprintf("%s.csv", arrName));
    fileId = fopen(fname, 'w+');

    header = "FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7, F8,T3,T4,T5,T6,FZ,CZ,PZ";
    fprintf(fileId, '%s\n', header);

    sizeArr = size(arrData);
    for colIdx = 1 : sizeArr(2)
        separator = '';
        for rowIdx = 1 : sizeArr(1)
            dataVal = arrData(rowIdx, colIdx);
            fprintf(fileId, '%s%d', separator, dataVal);
            separator = ',';
        end
        fprintf(fileId, '\n');
    end

    fclose(fileId);
end
