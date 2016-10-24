
% expID = 'hg-256-res-64-h36m-hgfix-w0'; mode = 1;
% expID = 'hg-256-res-64-h36m-hgfix-w0.001'; mode = 1;
% expID = 'hg-256-res-64-h36m-hgfix-w1'; mode = 1;

% split = 'train';
% split = 'val';

% set vis root
vis_root = ['./outputs/vis_h36m/' expID '/' split '/'];
makedir(vis_root);

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
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');

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
dataset_hg = h36m(opt, split);

% sample frames
run = 1:101:size(ind2sub,1);

% init figure
figure(1);
set(gcf,'Position',[2 26 1830 330]);
clear hi hh hs1 hs2 hr1 hr2

% load libraries
libimg = img();

fprintf('visualizing penn predictions ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    sid = ind2sub(i,1);
    aid = ind2sub(i,2);
    bid = ind2sub(i,3);
    fid = ind2sub(i,4);
    vis_file = [vis_root sprintf('%02d_%02d_%1d_%04d.png',sid,aid,bid,fid)];
    if exist(vis_file,'file')
        continue
    end
    % show image
    cam = mod(i-1,4)+1;
    im_file = sprintf('data/h36m/frames/%02d/%02d/%1d/%1d_%04d.jpg', ...
        ind2sub(i,1),ind2sub(i,2), ...
        ind2sub(i,3),cam,ind2sub(i,4));
    im = imread(im_file);
    if exist('hi','var')
        delete(hi);
    end
    hi = subplot('Position',[0.00+0/6 0.00 1/6-0.00 1.00]);
    imshow(im); hold on;
    % draw heatmap
    [im, ~, ~, ~] = dataset_hg.get(i,cam);
    im = permute(im, [2 3 1]);
    hm_dir = ['./exp/h36m/' expID '/hmap_' split '/'];
    hm_file = [hm_dir num2str(i,'%05d') '.mat'];
    if exist(hm_file,'file')
        hm = load(hm_file);
        hm = hm.hmap;
    end
    if exist('hh','var')
        delete(hh);
    end
    if exist('hm','var')
        hh = subplot('Position',[0.00+1/6 0.00 1/6-0.00 1.00]);
        inp64 = imresize(double(im),[64 64]) * 0.3;
        colorHms = cell(size(hm,1),1);
        for j = 1:size(hm,1)
            colorHms{j} = libimg.colorHM(squeeze(hm(j,:,:)));
            colorHms{j} = colorHms{j} * 255 * 0.7 + permute(inp64,[3 1 2]);
        end
        totalHm = libimg.compileImages(colorHms, 4, 4, 64);
        totalHm = permute(totalHm,[2 3 1]);
        totalHm = uint8(totalHm);
        imshow(totalHm);
    end
    % show 3D skeleton in camera coordinates
    for j = 1:2
        if j == 1
            if exist('hs1','var')
                delete(hs1);
            end
            hs1 = subplot('Position',[0.03+2/6 0.07 1/6-0.04 0.93]);
        end
        if j == 2
            if exist('hs2','var')
                delete(hs2);
            end
            hs2 = subplot('Position',[0.02+3/6 0.07 1/6-0.03 0.93]);
        end
        set(gca,'fontsize',6);
        pred = permute(repos(i,:,:),[2 3 1]);
        pred = pred + repmat(permute(trans(i,:),[2 1]),[1 size(pred,2)]);
        V = pred;
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',2);
        end
        pose = permute(poses(i,:,:),[2 3 1]);
        V = pose;
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',2);
            set(hpos(k+1),'color',clr{k});
        end
        minx = -1500; maxx = 1500;
        miny =     0; maxy = 6500;
        minz = -1500; maxz = 1500;
        axis([minx maxx miny maxy minz maxz]);
        set(gca,'ZTick',-2000:400:2000);
        set(gca,'ZDir','reverse');
        if j == 1
            view([6,10]);
        end
        if j == 2
            view([85,10]);
        end
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
    end
    % show 3D skeleton relative to center
    for j = 1:2
        if j == 1
            if exist('hr1','var')
                delete(hr1);
            end
            hr1 = subplot('Position',[0.02+4/6 0.07 1/6-0.035 0.93]);
        end
        if j == 2
            if exist('hr2','var')
                delete(hr2);
            end
            hr2 = subplot('Position',[0.02+5/6 0.07 1/6-0.035 0.93]);
        end
        set(gca,'fontsize',6);
        pred = permute(repos(i,:,:),[2 3 1]);
        pred(1,:) = pred(1,:) - 500;
        pred(3,:) = pred(3,:) - 500;
        V = pred;
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',3);
        end
        pose = permute(poses(i,:,:),[2 3 1]);
        pose = pose - repmat(mean(pose,2),[1 size(pose,2)]);
        pose(1,:) = pose(1,:) + 500;
        pose(3,:) = pose(3,:) + 500;
        V = pose;
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',3);
            set(hpos(k+1),'color',clr{k});
        end
        minx = -1000; maxx = 1000;
        miny = -1000; maxy = 1000;
        minz = -1000; maxz = 1000;
        axis([minx maxx miny maxy minz maxz]);
        set(gca,'ZTick',-1000:200:1000);
        set(gca,'ZDir','reverse');
        if j == 1
            view([6,10]);
        end
        if j == 2
            view([85,10]);
        end
    end
    % save figure
    set(gcf,'PaperPositionMode','auto');
    print(gcf,vis_file,'-dpng','-r0');
end
fprintf('done.\n');

close;