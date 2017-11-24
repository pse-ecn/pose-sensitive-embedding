function result = readFileByLines(path)

fid=fopen(path);
cellResult = textscan(fid,'%s');
cellResult = cellResult{:};
result = string(cellstr(cellResult));
fclose(fid);