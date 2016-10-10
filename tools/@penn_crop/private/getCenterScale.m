function [ center, scale ] = getCenterScale( obj, im )

assert(ndims(im) == 3);
w = size(im,2);
h = size(im,1);
x = (w+1)/2;
y = (h+1)/2;
scale = max(w,h)/200;
center = [x, y]; 

end

