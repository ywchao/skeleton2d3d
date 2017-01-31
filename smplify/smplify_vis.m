split = 'val';

% init dataset and convert to penn format
dataset = load(['./data/h36m/' split '.mat']);

% ind = [701 13363 13199 15832 13351 13347 13241 6118 13343 13203];

interval = 101;

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');

pose_root = './smplify/smplify_public/results/h36m/';
save_root = './smplify/smplify_public/vis/h36m/';
makedir(save_root);

figure(1);

set(gcf,'Position',[2 26 1440 480]);

% reading annotations
fprintf('evaluating smplify on h36m ... \n');
for i = 1:interval:size(dataset.ind2sub,1)
    tic_print(sprintf('%05d/%05d\n',i,size(dataset.ind2sub,1)));
    
    sid = dataset.ind2sub(i,1);
    aid = dataset.ind2sub(i,2);
    bid = dataset.ind2sub(i,3);
    fid = dataset.ind2sub(i,4);
    cam = mod(i-1, 4) + 1;
    
    % skip if vis file exists
    save_file = sprintf('%s/%02d_%02d_%1d_%04d_%1d.png',save_root,sid,aid,bid,fid,cam);
    if exist(save_file,'file')
        continue
    end
    
    clf;
    
    im_file = sprintf('data/h36m/frames/%02d/%02d/%1d/%1d_%04d.jpg', ...
        dataset.ind2sub(i,1), ...
        dataset.ind2sub(i,2), ...
        dataset.ind2sub(i,3), ...
        cam, ...
        dataset.ind2sub(i,4));
    subplot(1,3,1);
    set(gca,'Position',[0 0 1/3 1]);
    im = imread(im_file);
    imshow(im); hold on;
    
    % load pred shape
    im_file = sprintf('smplify/smplify_public/results/h36m/%05d.png', i);
    subplot(1,3,2);
    set(gca,'Position',[1/3 0 1/3 1]);
    im = imread(im_file);
    imshow(im); hold on;
    
    % load pred skeleton
    pose_file = [pose_root sprintf('%05d.mat',i)];
    pose = load(pose_file);
    pred = pose.Jtr * 1000;
    joints = [16,18,17,20,19,22,21,3,2,6,5,9,8];
    pred = pred(joints,:);
    cntr = mean(pred,1);
    pred = pred - repmat(cntr,[size(pred,1) 1]);    
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_ = zeros(17,3);
    pred_(joints,:) = pred;
    pred_(1,:) = (pred(8,:) + pred(9,:))/2;
    pred_(8,:) = (pred(2,:) + pred(3,:) + pred(8,:) + pred(9,:))/4;
    pred_(9,:) = (pred(1,:) + pred(2,:) + pred(3,:))/3;
    pred_(11,:) = pred(1,:);

    subplot(1,3,3);
    set(gca,'Position',[2/3+0.05 0+0.07 1/3-0.08 0.93]);
    V = pred_';
    V([2 3],:) = V([3 2],:);
    showPose(V,posSkel);
    minx = -1000; maxx = 1000;
    miny = -1000; maxy = 1000;
    minz = -1000; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'ZDir','reverse');
    view([6,10]);
    
    % save vis to file
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpng','-r0');
end
fprintf('done.\n');

close;
