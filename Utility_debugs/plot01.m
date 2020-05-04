% plot figures for paper draft

wsi_path='../../../data/tcga_coad_slide/tcga_coad/quality_uncertain/';
close all;clc;
addpath(genpath('Y:\projects\xhm_code_repos\matlab_repository\toolboxes\openslide-matlab-master\'));
addpath(genpath('Y:\projects\xhm_code_repos\matlab_repository\my_codes\'));


magCoarse=2.5;
magFine=5;
mag10=20;
mag20=20;
ppt=0.8;         % above this threshold is selected as segmented regions
debug=0;
LBP_baseline=false; % if false not computing LBP baseline texture features for comparison

thrWhite=210;
numc=50;
tileSize=round([1013,1013]./16);


heatmap=imread('../Tumor_detector_stomach_colon/TCGA-NH-A50T-01Z-00-DX1_gray.png');

imgs=dir(fullfile(wsi_path,'*.svs'));

for im=13:numel(imgs)
    file1=fullfile(wsi_path,imgs(im).name);
    fprintf('filename=%s\n%d',file1,im);
    slidePtr=openslide_open(file1);
    [mppX,mppY,width,height,numberOfLevels,...
        downsampleFactors,objectivePower]=openslide_get_slide_properties(slidePtr);
    mag=objectivePower./round(downsampleFactors);
    
    %1) read magCoarse image
    RGB=wsi_read(slidePtr,objectivePower,downsampleFactors,width,height,magCoarse);
    h=size(RGB,1);
    w=size(RGB,2);
    
    %2) tiling
    bwTissue=rgb2gray(RGB)<240;
    fun=@(x)sum(sum(x.data))/(tileSize(1)*tileSize(2));
    rs=1;
    cs=1;
    re=h;
    ce=w;
    BTis=blockproc(bwTissue(rs:re,cs:ce),tileSize,fun);   %% tissue binary mask
    B=(BTis>0.5);
   
   
    yy=[rs,rs+tileSize(2):tileSize(2):re,re];           %% row direction
    xx=[cs,cs+tileSize(1):tileSize(1):ce,ce];           %% column direction
    [ry,cx]=find(B);
    top_left=[];
    bottom_right=[];
    top_left=[top_left;yy(ry)',xx(cx)'];              %% the first column: row; the second column: column; top-left point position
    bottom_right=[bottom_right;yy(ry+1)',xx(cx+1)'];  %% the first column: row; the second column: column; bottom-right point position
    xu_debugShownTiles_V2(RGB,bwTissue,top_left,tileSize);
    
%     % overlap heatmap
%     h=size(RGB,1);
%     w=size(RGB,2);
%     
%     % use exp to enforce contrast
%     heatmap = imresize(double(heatmap(:,:,1)), [h, w]);
%     
%     heatmap=double(heatmap(:,:,1))/255.0;
%     figure;
%     ax1=axes;
%     imagesc(RGB)
%     ax2=axes;
%     imagesc(ax2,heatmap,'alphadata',heatmap>0);
%     colormap(ax2,'hot');
%     caxis(ax2,[min(nonzeros(heatmap)) max(nonzeros(heatmap))]);
%     ax2.Visible = 'off';
%     linkprop([ax1 ax2],'Position');
%     colorbar;
%     %heatmap = exp(heatmap * 1.0);
%     
%     % choose colormap
%     colormap('parula')
%     
%     % resize the heatmap to the image size
%     
%     maxScores = max(max(heatmap));
%     minScores = min(min(heatmap));
%     
%     % normalized the heatmap: all values are in the range [0, 1]
%     normalizedHeatmap = (heatmap - minScores) / (maxScores - minScores);
%     
%     % display the image
%     figure(1), imagesc(RGB), axis image;
%     hold on
%     hImg = imagesc(255*normalizedHeatmap); 
%     axis off, caxis([0 255]);
%     
%     % the alpha parameter change the transparency of the heatmap
%     alpha = 0.3;
%     set(hImg, 'AlphaData', alpha);
%     hold off
end
