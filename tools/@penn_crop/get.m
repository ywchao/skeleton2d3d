function [ input, target, center, scale ] = get( obj, idx )
% This does not produce the exact output as the get() in penn-crop.lua
%   1. input is of type uint8 with values in range [0 255]

% Load image
im = loadImage(obj, idx);

% Get center and scale
[center, scale] = getCenterScale(obj, im);

% Transform image
% TODO: following the implement of penn-crop.lua
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(im, 1) > size(im, 2)
    outputSize = [obj.inputRes NaN];
else
    outputSize = [NaN obj.inputRes];
end
im = imresize(im, outputSize);
% pad zeros
input = uint8(zeros(obj.inputRes,obj.inputRes,3));
if size(im,1) > size(im,2)
    ul = [1 round((obj.inputRes-size(im,2))/2)+1];
else
    ul = [round((obj.inputRes-size(im,1))/2)+1 1];
end
input(ul(1):ul(1)+size(im,1)-1, ul(2):ul(2)+size(im,2)-1, :) = im;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate target
pts = squeeze(obj.part(idx,:,:));
vis = squeeze(obj.visible(idx,:));
target = zeros(obj.inputRes,obj.inputRes,size(pts,1));
for i = 1:size(pts,1)
    if vis(i)
        target(:,:,i) = obj.img.drawGaussian(target(:,:,i),obj.img.transform(pts(i,:), center, scale, 0, obj.outputRes),2);
    end
end
target = permute(target,[3 1 2]);

end

