clc;
clear;

% list folder pathes containing .bin files
[folder, binlog1, binnum1, stop] = srchfile({'program','focus'});
if stop || ~max(binlog1(:))==min(binlog1(:))
    return
end


   %% 
 for  k = 1:numel(folder) 
     tic
    disp(['Now processing ',folder{k}]);
    % Make path for saving data.
    slush = find(ismember(folder{k},'\'), 2, 'last');   % find the last / before cell name
    cellname = folder{k}(slush(1)+1:slush(2)-1);  % extract cell name from folder path
    conc = folder{k}(slush(2)+1:end);
%     cellname = ['MCF7GFP',conc];
    [ImgPath, ~, UnsegPath, ~, ~, ~] = MakePath(1,0,1,0,0,0,folder{k},cellname,[]);
    
    % Create a folder for bad images
    badseg = fullfile(ImgPath,'badimg',filesep);
    matfiles = fullfile(ImgPath,'matfiles',filesep);
    badunseg = fullfile(UnsegPath,'badimg',filesep);
    tiledimg = fullfile(ImgPath,'tiled',filesep);
%     unseg48 = [UnsegPath,'/unseg48/'];
    if ~isdir(badseg)
        mkdir(badseg);
    end
    if ~isdir(badunseg)
       mkdir(badunseg);
    end
    if ~isdir(matfiles)
       mkdir(matfiles);
    end
    if ~isdir(tiledimg)
       mkdir(tiledimg);
    end
 % list data files.
    binname = ls(fullfile(folder{k}, '*Img.bin'));
    logname = ls(fullfile(folder{k}, '*log.mat'));

%%    

stat_area = zeros(1242,size(logname,1));

for j = 1:size(logname,1)
    lognamej = fullfile(folder{k}, logname(j,:));
    logfile = importdata(lognamej); % import a log file
    v2struct(logfile);  % Toolbox required (Pack & Unpack variables to & from structures)
    % import a bin file
    binnamej = fullfile(folder{k}, binname(j,:));
    fileID = fopen(binnamej,'r','l');
    imgdata = fread(fileID,'int8');
    fclose(fileID);

% % Sort images
imgdata = reshape(imgdata, [], FrameNum);

filename1 = sprintf('Img%04d',j-1);   % file name for each image

%% Image recovery

    parfor i = 1:FrameNum
    
        [Image, RepRate] = ImgRecoveryhisto(imgdata(:,i), SampleRate, -1, 10);
        imagefile = sprintf([filename1, '_', cellname, '%02d'],i);
        
        
        ResFlow = FlowSpeed / RepRate * 1e6;	% Size of one pixel in the flowing direction (um)
        ResScan = SpotRes / SampleRate * 1e9;	% Size of one pixel in the scanning direction (um)
        
        % Low-pass filter & Reshape image & normalization of the intensity within [0 1] for imwrite.
        Image = LPResize2(Image, 50, 0.3, ResFlow, ResScan);
        Img = Image;    % spare the unsegmented image
        
       % Edge detection and outline the cell.
        [~,mask] = segactivecontour(Image,3)  ;
        Image(~mask) = 0;
        stat = regionprops(mask,'Area');

     if     ~isempty({stat.Area})
     stat_area(i,j)=cell2mat({stat.Area}).* ResScan .* ResScan;
     end

      % Evaluate the image quality.
        if max(mask(:))<1
            imwrite(Img, [badunseg, imagefile, '.tif']);   % save the unsegmented image
        else



             Img2 = boundingcrop(Img, mask, 164, 164, 0);
             Image = boundingcrop(Image, mask, 160, 160, 1);
             imwrite(Image, [ImgPath, imagefile, '.tif']);   % save the image
             imwrite(Img2, [UnsegPath, imagefile, '.tif']);   % save the unsegmented image

        end
             
    end

end

     
     areaname = sprintf([cellname,conc,'area']);
     save([areaname,'.mat'],'stat_area');

foldertime = toc;
disp(['Time elapsed: ', num2str(foldertime), 'sec.']);
 end 