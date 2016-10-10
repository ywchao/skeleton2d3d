classdef penn_crop
    properties (Access = private)
        split
        dir
        ind2sub
        visible
        part
        inputRes
        outputRes
        img
    end
    methods
        % constructor
        function obj = penn_crop(opt, split)
            obj.split = split;
            obj.dir = fullfile(opt.data, 'frames');
            assert(exist(obj.dir,'dir') == 7, ['directory does not exist: ' obj.dir]);
            % load annotation
            annot_file = fullfile(opt.data, [split '.h5']);
            obj.ind2sub = permute(hdf5read(annot_file,'ind2sub'),[2 1]);
            obj.visible = permute(hdf5read(annot_file,'visible'),[2 1]);
            obj.part = permute(hdf5read(annot_file,'part'),[3 2 1]);
            % get input and output resolution
            obj.inputRes = opt.inputRes;
            obj.outputRes = opt.inputRes;
            % load lib
            obj.img = img();
        end
        
        % get image path
        pa = imgpath(obj, idx);
        
        % load image
        im = loadImage(obj, idx);
        
        % get center and scale
        [center, scale] = getCenterScale(obj, im);
        
        % get dataset size
        out = size(obj);
        
        [input, target, center, scale] = get(obj, idx);
    end
end