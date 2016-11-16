
expID = 'hg-256-res-64-hg-pred'; mode = 1;

split = 'train';

% set save root
save_root = './outputs/figures/pull_fig/';
makedir(save_root);

% load annotations
ind2sub = hdf5read(['./data/penn-crop/' split '.h5'],'ind2sub');
visible = hdf5read(['./data/penn-crop/' split '.h5'],'visible');
part = hdf5read(['./data/penn-crop/' split '.h5'],'part');
ind2sub = permute(ind2sub,[2 1]);
visible = permute(visible,[2 1]);
part = permute(part,[3 2 1]);

% load predictions
preds = load(['./exp/penn-crop/' expID '/preds_' split '.mat']);
trans = preds.trans;
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
preds_ = zeros(size(preds.repos,1),17,3);
preds_(:,joints,:) = preds.repos;
preds_(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
preds_(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
preds_(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
preds_(:,11,:) = preds.repos(:,1,:);
preds = permute(preds_,[1,3,2]);
assert(size(preds,1) == size(ind2sub,1));

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

% get video ids for visualization
run = find(ismember(ind2sub(:,1),2034))';

sid = 20;
eid = 30;
interval = 3;
run = run(sid:interval:eid);

linewidth = linspace(1,4,numel(run));

% load image
ind = run(1);
sid = ind2sub(ind,1);
fid = ind2sub(ind,2);
im_file = ['data/penn-crop/frames/' num2str(sid,'%04d') '/' num2str(fid,'%06d') '.jpg'];
im = imread(im_file);
save_file = [save_root 'input.jpg'];
if ~exist(save_file,'file')
    set(gcf,'PaperPositionMode','auto');
    copyfile(im_file,save_file);
end

% show skeleton in 3d
for i = 1:numel(run)
    set(gcf,'Position',[0.00 0.00 560 560]);
    set(gca,'Position',[0.05 0.08 0.90 0.90]);
    set(gca,'fontsize',6);
    pred = permute(preds(run(i),:,:),[2 3 1]);
    pred(1,:) = pred(1,:) + i*100;
    pred(3,:) = pred(3,:) - i*40;
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',linewidth(i));
    end
    minx =  -400;  maxx =  800;
    miny =  -800;  maxy =  600;
    minz =  -600;  maxz =  700;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'ZDir','reverse');
    xlabel('');
    ylabel('');
    zlabel('');
    view([-25,10]);
end
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
set(gca,'zticklabel',[]);
save_file = [save_root 'skel3d.pdf'];
if ~exist(save_file,'file')
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end

% show skeleton in 2d
clf;
imshow(im);
fid_all = ind2sub(run,2);
clear ind2sub;
linewidth = linspace(1,3,numel(run));
for i = 1:numel(run)
    fid = fid_all(i);
    % load nn gt pose
    lb_file = ['data/penn-crop/labels/' sprintf('%04d.mat',sid)];
    anno = load(lb_file);
    pose = [anno.x(fid,:)' anno.y(fid,:)'];
    % convert to h36m format
    pose_ = zeros(17,2);
    pose_(joints,:) = pose;
    pose_(1,:) = (pose(8,:) + pose(9,:))/2;
    pose_(8,:) = (pose(2,:) + pose(3,:) + pose(8,:) + pose(9,:))/4;
    pose_(9,:) = (pose(1,:) + pose(2,:) + pose(3,:))/3;
    pose_(11,:) = pose(1,:);
    pose = pose_';
    vis = zeros(17,1);
    vis(joints) = anno.visibility(fid,:);
    vis(1) = anno.visibility(fid,8) && anno.visibility(fid,9);
    vis(8) = anno.visibility(fid,2) && anno.visibility(fid,3) && ...
        anno.visibility(fid,8) && anno.visibility(fid,9);
    vis(9) = anno.visibility(fid,1) && anno.visibility(fid,2) && anno.visibility(fid,3);
    vis(11) = anno.visibility(fid,1);
    % show2DPose(pose,pos2dSkel);
    padding = 0;
    pose = [pose zeros(1, padding)];  %#ok
    vals = bvh2xy(pos2dSkel, pose) ; % * 10
    connect = skelConnectionMatrix(pos2dSkel);
    indices = find(connect);
    [I, J] = ind2sub(size(connect), indices);
    hold on
    grid on
    for j = 1:length(indices)
        if vis(I(j)) == 0 || vis(J(j)) == 0
            continue
        end
        % modify with show part (3d geometrical thing)
        if strncmp(pos2dSkel.tree(I(j)).name,'L',1)
            c = 'r';
        elseif strncmp(pos2dSkel.tree(I(j)).name,'R',1)
            c = 'g';
        else
            c = 'b';
        end
        hl = line([vals(I(j),1) vals(J(j),1)],[vals(I(j),2) vals(J(j),2)],'Color',c);
        set(hl, 'linewidth', linewidth(i));
    end
    axis equal
end
save_file = [save_root 'skel2d.pdf'];
if ~exist(save_file,'file')
    set(gcf,'Position',[0.00 0.00 size(im,2) size(im,1)]);
    set(gca,'Position',[0.00 0.00 1.00 1.00]);
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf','-r0');
end

close;