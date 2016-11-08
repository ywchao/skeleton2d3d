
% add path and load data
shape_root = './shapeconvex/release/';
addpath([shape_root 'ssr']);
addpath([shape_root 'utils']);
shape_data = load([shape_root 'data/human/shapeDict.mat']);

% exp_name = 'hg-256-res-64-h36m-fthg';
% exp_name = 'hg-256-res-64-h36m-hg-pred';

split = 'val';

save_root = ['./shapeconvex/res_h36m_' exp_name '/' split '/'];
makedir(save_root);

% set opt and init dataset
opt.data = './data/h36m/';
opt.inputRes = 64;
opt.inputResHG = 256;
opt.hg = true;
opt.penn = true;
dataset = h36m(opt, split);

% start parpool
if ~exist('poolsize','var')
    poolobj = parpool();
else
    poolobj = parpool(poolsize);
end

% reading annotations
fprintf('processing shapeconvex on h36m ... \n');
parfor i = 1:dataset.size()
    % skip if vis file exists
    save_file = [save_root sprintf('%05d.mat',i)];
    if exist(save_file,'file')
        continue
    end
    fprintf('%05d/%05d  ',i,dataset.size());
    tt = tic;
    % load heatmap
    pred_file = sprintf('./exp/h36m/%s/eval_%s/%05d.mat',exp_name,split,i);
    pred = load(pred_file);
    pred = pred.eval;
    % convert from 13 to 15 joints
    X = convert_joint(pred(:,1)');
    Y = convert_joint(pred(:,2)');
    % convert joint order
    X = X(:,[15,9,11,13,8,10,12,14,1,3,5,7,2,4,6]);
    Y = Y(:,[15,9,11,13,8,10,12,14,1,3,5,7,2,4,6]);
    % compute 3d points
    W = normalizeS([X; Y]);
    S = ssr2D3D_wrapper(W,shape_data.B,'convex');
    % convert to single and save
    S = single(S);
    % save to file
    shapeconvex_save(save_file, S);
    time = toc(tt);
    fprintf('tot: %8.3f sec.  \n',time);
end
fprintf('done.\n');

% delete parpool
delete(poolobj);