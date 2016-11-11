
% exp_name = 'hg-256-res-64-h36m-hg-pred';        mode = 0;
% exp_name = 'hg-256-res-64-h36m-hgfix-w1';       mode = 1;
% exp_name = 'hg-256-res-64-h36m-fthg-hg-pred';   mode = 0;
% exp_name = 'hg-256-res-64-h36m-fthg-hgfix-w1';  mode = 1;

split = 'val';

interval = 101;

% set directories
if mode == 0
    pose_root = ['./shapeconvex/res_h36m_' exp_name '/' split '/'];
    save_root = ['./shapeconvex/vis_h36m_' exp_name '_final/' split '/'];
end
if mode == 1
    preds_file = ['./exp/h36m/' exp_name '/preds_' split '.mat'];
    preds = load(preds_file);
    save_root = ['./shapeconvex/vis_ours_h36m_' exp_name '/' split '/'];
end
makedir(save_root);

% init dataset
dataset = load(['./data/h36m/' split '.mat']);
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
ind2 = [];
for i = joints
    for j = 1:2
        ind2 = [ind2; (i-1)*2+j];  %#ok
    end
end
ind3 = [];
for i = joints
    for j = 1:3
        ind3 = [ind3; (i-1)*3+j];  %#ok
    end
end
dataset.coord_w = dataset.coord_w(:,ind3);
dataset.coord_c = dataset.coord_c(:,:,ind3);
dataset.coord_p = dataset.coord_p(:,:,ind2);

% get mean limb length
if mode == 0
    conn = [ 2  1; 3  1; 4  2; 5  3; 6  4; 7  5; 8  2; 9  3;10  8;11  9;12 10;13 11];
    coord_w = reshape(dataset.coord_w,[size(dataset.coord_w,1) 3 13]);
    d1 = coord_w(:,:,conn(:,1));
    d2 = coord_w(:,:,conn(:,2));
    mu = mean(sqrt(sum((d1 - d2).^2,2)),1);
    mu = permute(mu,[3 2 1]);
end
 
% set opt and init dataset
opt.data = './data/h36m/';
opt.inputRes = 64;
opt.inputResHG = 256;
opt.hg = true;
opt.penn = true;
dataset_mat = h36m(opt, split);

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

% set color for gt
clr = {'k','c','c','k','m','m','k','k','k','k','k','m','m','k','c','c'};

% load libraries
libimg = img();

% init figure
figure(1);
set(gcf,'Position',[2 26 1830 330]);
clear hi hh hs1 hs2 hr1 hr2

% reading annotations
fprintf('visualizing shapeconvex on h36m ... \n');
for i = 1:interval:size(dataset.ind2sub,1)
    tic_print(sprintf('%05d/%05d\n',i,size(dataset.ind2sub,1)));
    sid = dataset.ind2sub(i,1);
    aid = dataset.ind2sub(i,2);
    bid = dataset.ind2sub(i,3);
    fid = dataset.ind2sub(i,4);
    cam = mod(i-1,4)+1;
    % skip if vis file exists
    save_file = sprintf('%s/%02d_%02d_%1d_%04d_%1d.png',save_root,sid,aid,bid,fid,cam);
    if exist(save_file,'file')
        continue
    end
    % load 2D prediction
    pred_file = sprintf('./exp/h36m/%s/eval_%s/%05d.mat',exp_name,split,i);
    pred = load(pred_file);
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_ = zeros(17,2);
    pred_(joints,:) = pred.eval;
    pred_(1,:) = (pred.eval(8,:) + pred.eval(9,:))/2;
    pred_(8,:) = (pred.eval(2,:) + pred.eval(3,:) + pred.eval(8,:) + pred.eval(9,:))/4;
    pred_(9,:) = (pred.eval(1,:) + pred.eval(2,:) + pred.eval(3,:))/3;
    pred_(11,:) = pred.eval(1,:);
    pred = pred_;
    % draw 2D prediction
    if exist('hp','var')
        delete(hp);
    end
    hp = subplot('Position',[0.00+0/6 0.00 1/6-0.00 1.00]);
    im_file = sprintf('./data/h36m/frames/%02d/%02d/%1d/%1d_%04d.jpg',sid,aid,bid,cam,fid);
    im = imread(im_file);
    imshow(im); hold on;
    show2DPose(permute(pred,[2 1]),pos2dSkel);
    axis off;
    % load heatmap
    hmap_file = sprintf('./exp/h36m/%s/hmap_%s/%05d.mat',exp_name,split,i);
    hmap = load(hmap_file);
    hmap = hmap.hmap;
    % draw heatmap
    if exist('hh','var')
        delete(hh);
    end
    hh = subplot('Position',[0.00+1/6 0.00 1/6-0.00 1.00]);
    [input, ~, ~, ~, ~] = dataset_mat.get(i);
    input = permute(input, [2 3 1]);
    inp64 = imresize(double(input),[64 64]) * 0.3;
    colorHms = cell(size(hmap,1),1);
    for j = 1:size(hmap,1)
        colorHms{j} = libimg.colorHM(squeeze(hmap(j,:,:)));
        colorHms{j} = colorHms{j} * 255 * 0.7 + permute(inp64,[3 1 2]);
    end
    totalHm = libimg.compileImages(colorHms, 4, 4, 64);
    totalHm = permute(totalHm,[2 3 1]);
    totalHm = uint8(totalHm);
    imshow(totalHm);
    % load gt
    % [~, repos, ~, ~, ~] = dataset.get(i);
    cam = mod(i-1, 4) + 1;
    pose = reshape(dataset.coord_c(cam,i,:),[3 size(dataset.coord_c,3)/3])';
    pose = double(pose);
    cntr = mean(pose,1);
    repos = pose - repmat(cntr,[size(pose,1) 1]);
    if mode == 0
        % load predicted 3d pose
        pose_file = [pose_root sprintf('%05d.mat',i)];
        pose = load(pose_file);
        S = pose.S;
        % convert to penn format
        % joints = [9 13 10 14 11 15 12 5 2 6 3 7 4];
        % pred = S(:,joints)';
        joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
        pred = S(:,joints)';
        % scale pred to minimize error with length prior
        p1 = pred(conn(:,1),:);
        p2 = pred(conn(:,2),:);
        len = sqrt(sum((p1 - p2).^2,2));
        c = median(mu ./ len);
        pred = pred * c;
    end
    if mode == 1
        pred = preds.repos(i,:,:);
        pred = permute(pred,[2 3 1]);
    end
    % convert to h36m format
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    repos_ = zeros(17,3);
    repos_(joints,:) = repos;
    repos_(1,:) = (repos(8,:) + repos(9,:))/2;
    repos_(8,:) = (repos(2,:) + repos(3,:) + repos(8,:) + repos(9,:))/4;
    repos_(9,:) = (repos(1,:) + repos(2,:) + repos(3,:))/3;
    repos_(11,:) = repos(1,:);
    repos = repos_;
    pred_ = zeros(17,3);
    pred_(joints,:) = pred;
    pred_(1,:) = (pred(8,:) + pred(9,:))/2;
    pred_(8,:) = (pred(2,:) + pred(3,:) + pred(8,:) + pred(9,:))/4;
    pred_(9,:) = (pred(1,:) + pred(2,:) + pred(3,:))/3;
    pred_(11,:) = pred(1,:);
    pred = pred_;
    % show 3D skeleton
    for j = 1:4
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
        if j == 3
            if exist('hr1','var')
                delete(hr1);
            end
            hr1 = subplot('Position',[0.02+4/6 0.07 1/6-0.035 0.93]);
        end
        if j == 4
            if exist('hr2','var')
                delete(hr2);
            end
            hr2 = subplot('Position',[0.02+5/6 0.07 1/6-0.035 0.93]);
        end
        set(gca,'fontsize',6);
        V = pred';
        if j == 3 || j == 4
            V(1,:) = V(1,:) - 500;
            V(3,:) = V(3,:) - 500;
        end
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',3);
        end
        V = repos';
        if j == 3 || j == 4
            V(1,:) = V(1,:) + 500;
            V(3,:) = V(3,:) + 500;
        end
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
        if j == 1 || j == 3
            view([6,10]);
        end
        if j == 2 || j == 4
            view([85,10]);
        end
    end
    hs1 = subplot('Position',[0.03+2/6 0.07 1/6-0.04 0.93]);
    e = sqrt(sum((repos - pred).^2,2));
    e = mean(e);
    t = title(['error: ' num2str(e,'%.2f')]);
    set(t,'FontSize',10);
    % save vis to file
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpng','-r0');
end
fprintf('\n');

close;