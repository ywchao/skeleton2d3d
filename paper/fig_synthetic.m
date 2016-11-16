
rseed;

save_root = 'outputs/figures/synthetic/';
makedir(save_root);

db = H36MDataBase.instance();

posSkel = db.getPosSkel();
pos2dSkel = posSkel;
for i = 1 :length(pos2dSkel.tree)
    pos2dSkel.tree(i).posInd = [(i-1)*2+1 i*2];
end

data_file = './data/h36m/train.mat';
data = load(data_file);

Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');
[~, pos2dSkel] = Features{1}.select(zeros(0,64), pos2dSkel, 'body');

CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];

libimg = img();

% generate N samples
N = 30;

figure(1);

for n = 1:N
    % sample pose proj
    [cam, pind, pose, proj, depth, trial] = sample_pose_proj(data.coord_w);
    
    % draw skeleton and camera
    clf;
    set(gcf,'Position',[0.00 0.00 560 560]);
    set(gca,'Position',[0.05 0.08 0.90 0.90]);
    set(gca,'fontsize',6);
    hpos = showPose(pose,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',3);
    end
    ylabel('y');
    zlabel('z');
    minx = -2000; maxx = 2000;
    miny = -2000; maxy = 2000;
    minz = -1000; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    view([35,30]);
    CameraVertex(2:end,3) = cam.f(1) / 0.064;
    CVWorld = (cam.R'*CameraVertex')' + repmat(cam.T,[size(CameraVertex,1) 1]);
    hc = zeros(size(CameraVertex,1),1);
    for ind = 1:length(IndSetCamera)
        hc(ind) = patch( ...
            CVWorld(IndSetCamera{ind},1), ...
            CVWorld(IndSetCamera{ind},2), ...
            CVWorld(IndSetCamera{ind},3), ...
            [0.5 0.5 0.5]);
    end
    drawnow;
    set(gca,'XTick',-2000: 500:2000);
    set(gca,'YTick',-2000: 500:2000);
    set(gca,'ZTick',-1000: 500:1000);
    xlabel('');
    ylabel('');
    zlabel('');
    save_file = [save_root sprintf('%03d-3d.pdf',n)];
    if ~exist(save_file,'file')
        print(gcf,save_file,'-dpdf','-r0');
    end
    
    % draw projection
    clf;
    set(gca,'fontsize',6);
    hpos = show2DPose(proj,pos2dSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',5);
    end
    ylabel('y');
    axis([0 cam.res(1) 0 cam.res(2)]);
    axis ij;
    drawnow;
    box on;
    set(gcf,'Position',[0.00 0.00 560 560]);
    set(gca,'Position',[0.00 0.00 1.00 1.00]);
    set(gcf,'PaperPositionMode','auto');
    save_file = [save_root sprintf('%03d-2d.pdf',n)];
    if ~exist(save_file,'file')
        print(gcf,save_file,'-dpdf','-r0');
    end
    
    % create heatmaps
    hm = zeros(cam.res(1),cam.res(2),size(proj,2));
    for i = 1:size(proj,2)
        hm(:,:,i) = libimg.drawGaussian(hm(:,:,i),proj(:,i),2);
    end
    
    % save visualized 2d pose
    F = getframe(gca);
    I = frame2im(F);
    
    % draw heamap
    clf;
    hh = subplot('Position',[0.025+2/4 0.05 1/4-0.05 0.9]);
    inp64 = imresize(double(I),cam.res) * 0.3;
    colorHms = cell(size(proj,2),1);
    for i = 1:size(proj,2)
        colorHms{i} = libimg.colorHM(hm(:,:,i));
        colorHms{i} = colorHms{i} * 255 * 0.7 + permute(inp64,[3 1 2]);
    end
    colorHms = colorHms(joints);
    totalHm = libimg.compileImages(colorHms, 4, 4, 64);
    totalHm = permute(totalHm,[2 3 1]);
    totalHm = uint8(totalHm);
    imshow(totalHm);
    set(gcf,'Position',[0.00 0.00 size(totalHm,2) size(totalHm,1)]);
    set(gca,'Position',[0.00 0.00 1.00 1.00]);
    set(gcf,'PaperPositionMode','auto');
    save_file = [save_root sprintf('%03d-hmap.pdf',n)];
    if ~exist(save_file,'file')
        print(gcf,save_file,'-dpdf','-r0');
    end
end

close;