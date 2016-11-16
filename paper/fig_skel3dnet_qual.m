
expID = 'hg-256-res-64-h36m-fthg-hgfix-w1';

split = 'val';

% set save root
save_root = ['./outputs/figures/skel3dnet_qual_' expID '/'];
makedir(save_root);

% load annotations
anno = load(['./data/h36m/' split '.mat']);
ind2sub = anno.ind2sub;

% remove corrupted images
rm = [11,2,2];
for i = 1:size(rm,1)
    i1 = ind2sub(:,1) == rm(i,1);
    i2 = ind2sub(:,2) == rm(i,2);
    i3 = ind2sub(:,3) == rm(i,3);
    keep = find((i1+i2+i3) ~= 3);
    ind2sub = ind2sub(keep,:);
end

% load predictions
preds = load(['./exp/h36m/' expID '/preds_' split '.mat']);
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
repos = zeros(size(preds.repos,1),17,3);
repos(:,joints,:) = preds.repos;
repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
repos(:,11,:) = preds.repos(:,1,:);
repos = permute(repos,[1,3,2]);
assert(size(repos,1) == size(ind2sub,1));
trans = preds.trans;

% load groud truth
poses = zeros(size(preds.poses,1),17,3);
poses(:,joints,:) = preds.poses;
poses(:,1,:) = (preds.poses(:,8,:) + preds.poses(:,9,:))/2;
poses(:,8,:) = (preds.poses(:,2,:) + preds.poses(:,3,:) + preds.poses(:,8,:) + preds.poses(:,9,:))/4;
poses(:,9,:) = (preds.poses(:,1,:) + preds.poses(:,2,:) + preds.poses(:,3,:))/3;
poses(:,11,:) = preds.poses(:,1,:);
poses = permute(poses,[1,3,2]);
assert(size(poses,1) == size(ind2sub,1));

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
pos2dSkel = posSkel;
for i = 1 :length(pos2dSkel.tree)
    pos2dSkel.tree(i).posInd = [(i-1)*2+1 i*2];
end
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');
[~, pos2dSkel] = Features{1}.select(zeros(0,64), pos2dSkel, 'body');

% init camera
CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

% set color for gt
clr = {'k','c','c','k','m','m','k','k','k','k','k','m','m','k','c','c'};

% set opt and init dataset
opt.data = './data/h36m/';
opt.inputRes = 64;
opt.inputResHG = 256;
opt.hg = true;
opt.penn = true;
dataset = h36m(opt, split);

% load libraries
libimg = img();

% find(ismember(ind2sub,[5  2 1 1016],'rows'))  % 102
% find(ismember(ind2sub,[5 10 1  286],'rows'))  % 5253
% find(ismember(ind2sub,[6  3 2 1646],'rows'))  % 10909
% find(ismember(ind2sub,[6 11 1 1886],'rows'))  % 13636

% find(ismember(ind2sub,[6  8 2  446],'rows'))  % 12626
% find(ismember(ind2sub,[6  2 2 1126],'rows'))  % 10303
% find(ismember(ind2sub,[6 12 1  866],'rows'))  % 14141
% find(ismember(ind2sub,[5  6 1 2986],'rows'))  % 3233
% find(ismember(ind2sub,[5  9 2  126],'rows'))  % 4849
% find(ismember(ind2sub,[5  9 2 1136],'rows'))  % 4950

clear i;

% i = 102;
% i = 5253;
% i = 10909;
% i = 13636;

sid = ind2sub(i,1);
aid = ind2sub(i,2);
bid = ind2sub(i,3);
fid = ind2sub(i,4);
cam = mod(i-1,4)+1;

% show image
save_file = [save_root sprintf('%02d-%02d-%1d-%1d-%04d-im.pdf',sid,aid,bid,cam,fid)];
if ~exist(save_file,'file')
    clf;
    im_file = sprintf('data/h36m/frames/%02d/%02d/%1d/%1d_%04d.jpg', ...
        ind2sub(i,1),ind2sub(i,2), ...
        ind2sub(i,3),cam,ind2sub(i,4));
    im = imread(im_file);
    imshow(im);
    set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)]);
    set(gca,'Position',[0.00 0.00 1.00 1.00]);
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end

% show heatmaps
save_file = [save_root sprintf('%02d-%02d-%1d-%1d-%04d-hmap.pdf',sid,aid,bid,cam,fid)];
if ~exist(save_file,'file')
    clf;
    [input, ~, ~, ~, ~] = dataset.get(i);
    input = permute(input, [2 3 1]);
    hm_dir = ['./exp/h36m/' expID '/hmap_' split '/'];
    hm_file = [hm_dir num2str(i,'%05d') '.mat'];
    hm = load(hm_file);
    hm = hm.hmap;
    inp64 = imresize(double(input),[64 64]) * 0.3;
    colorHms = cell(size(hm,1),1);
    for j = 1:size(hm,1)
        colorHms{j} = libimg.colorHM(squeeze(hm(j,:,:)));
        colorHms{j} = colorHms{j} * 255 * 0.7 + permute(inp64,[3 1 2]);
    end
    totalHm = libimg.compileImages(colorHms, 4, 4, 64);
    totalHm = permute(totalHm,[2 3 1]);
    totalHm = uint8(totalHm);
    imshow(totalHm);
    set(gcf,'Position',[0.00 0.00 size(totalHm,2) size(totalHm,1)]);
    set(gca,'Position',[0.00 0.00 1.00 1.00]);
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end

% show prediction 3D skeleton
save_file = [save_root sprintf('%02d-%02d-%1d-%1d-%04d-pred.pdf',sid,aid,bid,cam,fid)];
if ~exist(save_file,'file')
    clf;
    set(gcf,'Position',[0.00 0.00 560 560]);
    set(gca,'Position',[0.05 0.08 0.90 0.90]);
    set(gca,'fontsize',6);
    pred = permute(repos(i,:,:),[2 3 1]);
    pred = pred + repmat(permute(trans(i,:),[2 1]),[1 size(pred,2)]);
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',5);
    end
    minx = -1000; maxx = 1000;
    miny =     0; maxy = 6500;
    minz = -1500; maxz =  500;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'XTick',-1000: 200:1000);
    set(gca,'YTick',    0:1000:6500);
    set(gca,'ZTick',-1500: 200: 500);
    set(gca,'ZDir','reverse');
    view([6,10]);
    CVWorld = CameraVertex;
    CVWorld(:,[2 3]) = CVWorld(:,[3 2]);
    hc = zeros(size(CameraVertex,1),1);
    for ind = 1:length(IndSetCamera)
        hc(ind) = patch( ...
            CVWorld(IndSetCamera{ind},1), ...
            CVWorld(IndSetCamera{ind},2), ...
            CVWorld(IndSetCamera{ind},3), ...
            [0.5 0.5 0.5]);
    end
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end

% show prediction 3D skeleton with gt
save_file = [save_root sprintf('%02d-%02d-%1d-%1d-%04d-pred-gt.pdf',sid,aid,bid,cam,fid)];
if ~exist(save_file,'file')
    clf;
    set(gcf,'Position',[0.00 0.00 560 560]);
    set(gca,'Position',[0.05 0.08 0.90 0.90]);
    set(gca,'fontsize',6);
    pred = permute(repos(i,:,:),[2 3 1]);
    pred(1,:) = pred(1,:) - 500;
    pred(3,:) = pred(3,:) - 500;
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',5);
    end
    pose = permute(poses(i,:,:),[2 3 1]);
    pose = pose - repmat(mean(pose,2),[1 size(pose,2)]);
    pose(1,:) = pose(1,:) + 500;
    pose(3,:) = pose(3,:) + 500;
    V = pose;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',5);
        set(hpos(k+1),'color',clr{k});
    end
    minx = -1000; maxx = 1000;
    miny = -1000; maxy = 1000;
    minz = -1000; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'XTick',-1000:1000:1000);
    set(gca,'YTick',-1000: 200:1000);
    set(gca,'ZTick',-1000: 200:1000);
    set(gca,'ZDir','reverse');
    view([85,10]);
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end

close;