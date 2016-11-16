
H36MDataBase.instance();

Features{1} = H36MPose3DPositionsFeature();

% set parameters
part = 'body';
samp = 10;

% validation set
s = 1;
a = 13;
b = 2;
p = 800;

figure(1);
set(gcf,'Position',[2 26 700 720]);
clear hs hp

CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

data_file = './data/h36m/val.mat';

c = 1;
Sequence = H36MSequence(s, a, b, c);
F = H36MComputeFeatures(Sequence, Features);
Subject = Sequence.getSubject();
posSkel = Subject.getPosSkel();
[pose, posSkel] = Features{1}.select(F{1}, posSkel, part);

Camera = Sequence.getCamera();

P = reshape(pose(p,:),[3 numel(pose(p,:))/3])';
N = size(P,1);
X = Camera.R*(P'-Camera.T'*ones(1,N));
    
% draw skeleton and camera
if exist('hs','var')
    delete(hs);
end
set(gca,'Position',[0.1 0.05 0.85 0.95]);
set(gca,'fontsize',10);
V = X;
V([2 3],:) = V([3 2],:);
showPose(V,posSkel);
minx = -1500; maxx = 1500;
miny =     0; maxy = 6500;
minz = -1500; maxz = 1500;
axis([minx maxx miny maxy minz maxz]);
set(gca,'ZDir','reverse');
view([35,30]);
CameraVertex(2:end,3) = Camera.f(1);
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
drawnow;

save_dir = './outputs/figures/';
makedir(save_dir);
save_file = [save_dir 'repos_trans_focal.pdf'];
if ~exist(save_file,'file')
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpdf');
end
