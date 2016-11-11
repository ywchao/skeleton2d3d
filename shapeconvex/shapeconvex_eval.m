
% exp_name = 'hg-256-res-64-h36m-hg-pred';        mode = 0;
% exp_name = 'hg-256-res-64-h36m-hgfix-w1';       mode = 1;
% exp_name = 'hg-256-res-64-h36m-fthg-hg-pred';   mode = 0;
% exp_name = 'hg-256-res-64-h36m-fthg-hgfix-w1';  mode = 1;

split = 'val';

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

% set directories
if mode == 0
    pose_root = ['./shapeconvex/res_h36m_' exp_name '/' split '/'];
end
if mode == 1
    preds_file = ['./exp/h36m/' exp_name '/preds_' split '.mat'];
    preds = load(preds_file);
end

err = zeros(1+13,size(dataset.ind2sub,1));
sca = zeros(size(dataset.ind2sub,1),1);

% reading annotations
fprintf('evaluating shapeconvex on h36m ... \n');
for i = 1:size(dataset.ind2sub,1)
    tic_print(sprintf('%05d/%05d\n',i,size(dataset.ind2sub,1)));
    tt = tic;
    % load gt
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
        c = 1;
    end
    
    e = sqrt(sum((repos - pred).^2,2));
    err(:,i) = [mean(e); e];
    sca(i) = c;
end
fprintf('done.\n');

disp(mean(err,2));

% [~, ii] = sort(err(1,:),'descend');
% for i = 1:10
%     sid = dataset.ind2sub(ii(i),1);
%     aid = dataset.ind2sub(ii(i),2);
%     bid = dataset.ind2sub(ii(i),3);
%     fid = dataset.ind2sub(ii(i),4);
%     cam = mod(ii(i)-1, 4) + 1;
%     fprintf('%02d %02d %1d %04d %1d\n',sid,aid,bid,fid,cam);
% end