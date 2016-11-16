
split = 'test';

im_id = 99;

save_dir = './outputs/figures/';
makedir(save_dir);

% set opt and init dataset
opt.data = './data/penn-crop';
opt.inputRes = 64;
opt.inputResHG = 256;

% get input and projection
opt.hg = true;
dataset = penn_crop(opt, split);
[input, proj, ~, ~] = dataset.get(im_id);
proj = proj';

% get heatmap
opt.hg = false;
dataset = penn_crop(opt, split);
dataset.get(im_id);
[hmap, ~, ~, ~] = dataset.get(im_id);
hmap = permute(hmap,[2 3 1]);

% draw input
figure(1);
imshow(input);

save_file = [save_dir 'im_proj_1.pdf'];
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf');
end

% draw heatmap
figure(2);
libimg = img();
inp64 = imresize(double(input),[opt.inputRes opt.inputRes]) * 0.3;
colorHms = cell(size(proj,2),1);
for i = 1:size(proj,2)
    colorHms{i} = libimg.colorHM(hmap(:,:,i));
    colorHms{i} = colorHms{i} * 255 * 0.7 + permute(inp64,[3 1 2]);
end
totalHm = libimg.compileImages(colorHms, 4, 4, opt.inputRes);
totalHm = permute(totalHm,[2 3 1]);
totalHm = uint8(totalHm);
imshow(totalHm);

save_file = [save_dir 'im_proj_2.pdf'];
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf');
end

% convert projection
db = H36MDataBase.instance();

posSkel = db.getPosSkel();
pos2dSkel = posSkel;
for i = 1 :length(pos2dSkel.tree)
    pos2dSkel.tree(i).posInd = [(i-1)*2+1 i*2];
end
Features{1} = H36MPose3DPositionsFeature();
[~, pos2dSkel] = Features{1}.select(zeros(0,64), pos2dSkel, 'body');

joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
proj_ = zeros(2,17);
proj_(:,joints) = proj;
proj_(:,1) = (proj(:,8) + proj(:,9))/2;
proj_(:,8) = (proj(:,2) + proj(:,3) + proj(:,8) + proj(:,9))/4;
proj_(:,9) = (proj(:,1) + proj(:,2) + proj(:,3))/3;
proj_(:,11) = proj(:,1);
proj = proj_;

% draw projection
figure(3);
set(gca,'fontsize',6);
show2DPose(proj,pos2dSkel);
ylabel('y');
axis([0 opt.inputRes 0 opt.inputRes]);
axis ij;

save_file = [save_dir 'im_proj_3.pdf'];
if ~exist(save_file,'file')
    print(gcf,save_file,'-dpdf');
end

close all;