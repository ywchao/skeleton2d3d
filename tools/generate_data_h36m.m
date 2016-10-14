
H36MDataBase.instance();

Features{1} = H36MPose3DPositionsFeature();

% set parameters
part = 'body';
samp = 10;

% training set
S = [1 7 8 9 11];

data_file = './data/h36m/train.mat';
if ~exist(data_file,'file')
    fprintf('generate training data ... \n');
    ind2sub = int32(zeros(0,4));
    coord_w = single(zeros(0,51));
    coord_c = single(zeros(0,51));
    coord_p = single(zeros(0,34));
    focal = single(zeros(0,2));
    for s = S
        for a = 2:16
            for b = 1:2
                tt = tic;
                fprintf('  subject %02d, action %02d-%d',s,a,b);
                
                c = 1;
                Sequence = H36MSequence(s, a, b, c);
                F = H36MComputeFeatures(Sequence, Features);
                Subject = Sequence.getSubject();
                posSkel = Subject.getPosSkel();
                [pose, posSkel] = Features{1}.select(F{1}, posSkel, part);
                
                sid = round((samp-1)/2)+1;
                ind = sid:samp:size(pose,1);
                pose = pose(ind,:);
                
                ind2sub = [ind2sub; [repmat([s a b],[numel(ind) 1]) ind']];  %#ok
                coord_w = [coord_w; pose];  %#ok
                
                Camera = Sequence.getCamera();
                for p = 1:size(pose,1)
                    P = reshape(pose(p,:),[3 numel(pose(p,:))/3])';
                    N = size(P,1);
                    X = Camera.R*(P'-Camera.T'*ones(1,N));
                    coord_c = [coord_c; reshape(X,[1 numel(X)])];  %#ok
                    
                    proj = Camera.project(Features{1}.toPositions(pose(p,:),posSkel));
                    coord_p = [coord_p; proj'];  %#ok
                    
                    focal = [focal; Camera.f];  %#ok
                end
                
                time = toc(tt);
                fprintf('  %7.2f sec.\n',time);
            end
        end
    end
    % need version -V6 for ilcomp
    save(data_file,'ind2sub','coord_w','coord_c','coord_p','focal','-V6');
    fprintf('done.\n');
end

% validation set
S = [5 6];

data_file = './data/h36m/val.mat';
if ~exist(data_file,'file')
    fprintf('generate validation data ... \n');
    ind2sub = int32(zeros(0,4));
    coord_w = single(zeros(0,51));
    coord_c = single(zeros(0,51));
    coord_p = single(zeros(0,34));
    focal = single(zeros(0,2));
    for s = S
        for a = 2:16
            for b = 1:2
                tt = tic;
                fprintf('  subject %02d, action %02d-%d',s,a,b);
                
                c = 1;
                Sequence = H36MSequence(s, a, b, c);
                F = H36MComputeFeatures(Sequence, Features);
                Subject = Sequence.getSubject();
                posSkel = Subject.getPosSkel();
                [pose, posSkel] = Features{1}.select(F{1}, posSkel, part);
                
                sid = round((samp-1)/2)+1;
                ind = sid:samp:size(pose,1);
                pose = pose(ind,:);
                
                ind2sub = [ind2sub; [repmat([s a b],[numel(ind) 1]) ind']];  %#ok
                coord_w = [coord_w; pose];  %#ok
                
                Camera = Sequence.getCamera();
                for p = 1:size(pose,1)
                    P = reshape(pose(p,:),[3 numel(pose(p,:))/3])';
                    N = size(P,1);
                    X = Camera.R*(P'-Camera.T'*ones(1,N));
                    coord_c = [coord_c; reshape(X,[1 numel(X)])];  %#ok
                    
                    proj = Camera.project(Features{1}.toPositions(pose(p,:),posSkel));
                    coord_p = [coord_p; proj'];  %#ok
                    
                    focal = [focal; Camera.f];  %#ok
                end
                
                time = toc(tt);
                fprintf('  %7.2f sec.\n',time);
            end
        end
    end
    % need version -V6 for ilcomp
    save(data_file,'ind2sub','coord_w','coord_c','coord_p','focal','-V6');
    fprintf('done.\n');
end
