function [ im ] = loadImage( obj, idx )

im = imread(fullfile(obj.dir, imgpath(obj, idx)));

end

