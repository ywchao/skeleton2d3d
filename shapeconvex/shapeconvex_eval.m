
% add path and load data
shape_root = './shapeconvex/release/';
addpath([shape_root 'ssr']);
addpath([shape_root 'utils']);
shape_data = load([shape_root 'data/human/shapeDict.mat']);

% exp_name = 'hg-256-res-64-h36m-fthg';
% exp_name = 'hg-256-res-64-h36m-hg-pred';

split = 'val';

% % set opt and init dataset
% opt.data = './data/h36m/';
% opt.inputRes = 64;
% opt.inputResHG = 256;
% opt.hg = true;
% opt.penn = true;
% dataset = h36m(opt, split);
% init dataset and convert to penn format
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
dataset.coord_w = dataset.coord_w(:,ind3,:);
dataset.coord_c = dataset.coord_c(:,:,ind3,:);
dataset.coord_p = dataset.coord_p(:,:,ind2,:);

% set directories
pose_root = ['./shapeconvex/res_h36m_' exp_name '/' split '/'];

% db = H36MDataBase.instance();
% posSkel = db.getPosSkel();
% Features{1} = H36MPose3DPositionsFeature();
% [~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');

% clr = {'k','c','c','k','m','m','k','k','k','k','k','m','m','k','c','c'};

err = zeros(1+13,1);

% reading annotations
fprintf('evaluating shapeconvex on h36m ... \n');
% for i = 1:dataset.size()
for i = 1:size(dataset.ind2sub,1)
    % tic_print(sprintf('%05d/%05d\n',i,dataset.size()));
    tic_print(sprintf('%05d/%05d\n',i,size(dataset.ind2sub,1)));
    tt = tic;
    % load predicted 3d pose
    pose_file = [pose_root sprintf('%05d.mat',i)];
    pose = load(pose_file);
    S = pose.S;
    % convert back to penn format
    joints = [9 13 10 14 11 15 12 5 2 6 3 7 4];
    pred = S(:,joints)';
    % load gt
    % [~, repos, ~, ~, ~] = dataset.get(i);
    cam = mod(i-1, 4) + 1;
    pose = reshape(dataset.coord_c(cam,i,:),[3 size(dataset.coord_c,3)/3])';
    pose = double(pose);
    cntr = mean(pose,1);
    repos = pose - repmat(cntr,[size(pose,1) 1]);
    % rescale pred to minimize mse with gt
    c = sum(repos(:) .* pred(:)) / sum(pred(:).^2);
    pred = pred * c;
    
    % joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    %
    % repos_ = zeros(17,3);
    % repos_(joints,:) = repos;
    % repos_(1,:) = (repos(8,:) + repos(9,:))/2;
    % repos_(8,:) = (repos(2,:) + repos(3,:) + repos(8,:) + repos(9,:))/4;
    % repos_(9,:) = (repos(1,:) + repos(2,:) + repos(3,:))/3;
    % repos_(11,:) = repos(1,:);
    % repos = repos_;
    %
    % pred_ = zeros(17,3);
    % pred_(joints,:) = pred;
    % pred_(1,:) = (pred(8,:) + pred(9,:))/2;
    % pred_(8,:) = (pred(2,:) + pred(3,:) + pred(8,:) + pred(9,:))/4;
    % pred_(9,:) = (pred(1,:) + pred(2,:) + pred(3,:))/3;
    % pred_(11,:) = pred(1,:);
    % pred = pred_;
    %
    % set(gca,'fontsize',6);
    % V = repos';
    % V([2 3],:) = V([3 2],:);
    % hpos = showPose(V,posSkel);
    % for k = 1:numel(hpos)-1
    %     set(hpos(k+1),'linewidth',2);
    % end
    % V = pred';
    % V([2 3],:) = V([3 2],:);
    % hpos = showPose(V,posSkel);
    % for k = 1:numel(hpos)-1
    %     set(hpos(k+1),'linewidth',2);
    %     set(hpos(k+1),'color',clr{k});
    % end
    % minx = -1000; maxx = 1000;
    % miny = -1000; maxy = 1000;
    % minz = -1000; maxz = 1000;
    % axis([minx maxx miny maxy minz maxz]);
    % set(gca,'ZTick',-1000:200:1000);
    % set(gca,'ZDir','reverse');
    % view([6,10]);
    
    e = sqrt(sum((repos - pred).^2,2));
    err = err + [mean(e); e];
end
fprintf('done.\n');

% disp(err/dataset.size());
disp(err/size(dataset.ind2sub,1));