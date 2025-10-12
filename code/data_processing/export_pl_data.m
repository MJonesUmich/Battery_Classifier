%% Export all "Data" tables inside a PLxx.mat to CSV files (SCRIPT version)
% Usage: set matFile below, then run this script.
clear; clc;

% 自动切换到脚本所在目录
cd(fileparts(mfilename('fullpath')));

%% === 1) 指定 .mat 文件（改这里） ===
matFile = 'PL13.mat';   % 可改为 'PL13.mat'

if ~isfile(matFile)
    error('File not found: %s', matFile);
end

%% === 2) 读取 .mat 的顶层变量（自动识别变量名） ===
S = load(matFile);              % 作为结构体加载
topVars = fieldnames(S);
if isempty(topVars)
    error('No variables found in %s', matFile);
end
topVarName = topVars{1};        % 取第一个顶层变量
dataCell   = S.(topVarName);    % 期望是 N x 3 的 cell：{Operation, StartDate, DataTable}

% 输出目录：以文件名为基础
[~, baseName, ~] = fileparts(matFile);
outDir = fullfile(pwd, [baseName '_csv']);
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% === 3) 遍历每一行 Operation，导出第3列 Data（table）为 CSV ===
nRows = size(dataCell, 1);
fprintf('Found %d operations in %s\n', nRows, baseName);

allExported = {};  % {Row, Operation, CSV_Path, NumRows}

for i = 1:nRows
    try
        % 读取 Operation 名（做文件名用；为空则用占位）
        opName = '';
        if i <= size(dataCell,1) && ~isempty(dataCell{i,1})
            opName = char(string(dataCell{i,1}));
        end
        if isempty(opName)
            opName = sprintf('operation_%02d', i);
        end
        % 清理非法文件名字符
        opName = regexprep(opName, '[^\w-]', '_');

        % 第3列为 Data 表
        T = [];
        if i <= size(dataCell,1)
            T = dataCell{i,3};
        end

        if istable(T)
            outFile = fullfile(outDir, sprintf('%s_%02d.csv', opName, i));
            writetable(T, outFile);
            fprintf('✅ Exported %s (%d rows)\n', outFile, height(T));
            allExported(end+1, :) = {i, opName, outFile, height(T)}; %#ok<AGROW>
        else
            % 不是 table（极少见）：打印类型供排查
            fprintf('⚠️  Row %d has no table data (class=%s) — skipped\n', i, class(T));
        end

    catch ME
        fprintf('❌ Error exporting row %d: %s\n', i, ME.message);
    end
end

%% === 4) 生成 manifest.csv（汇总导出情况） ===
if ~isempty(allExported)
    % 避免与 table 的行维度名 'Row' 冲突
    manifest = cell2table(allExported, ...
        'VariableNames', {'OpIndex', 'Operation', 'CSV_Path', 'NumRows'});
    writetable(manifest, fullfile(outDir, 'manifest.csv'));
    fprintf('Manifest saved: %s\n', fullfile(outDir, 'manifest.csv'));
end


fprintf('All done. CSV files saved to: %s\n', outDir);
