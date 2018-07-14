-- TODO: update following preprocess.py and penn.py

require 'hdf5'
require 'image'

local img = require 'lib/util/img'
local geometry = require 'lib/util/geometry'

local M = {}
local PennCropDataset = torch.class('skeleton2d3d.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  -- Load annotation
  annot_file = paths.concat(opt.data, split .. '.h5')
  self.ind2sub = hdf5.open(annot_file,'r'):read('ind2sub'):all()
  self.part = hdf5.open(annot_file,'r'):read('part'):all()
  -- Get input and output resolution
  self.inputRes = opt.inputResHG
  self.outputRes = opt.inputRes
  -- Get number of joints
  self.numPt = self.part:size(2)
  -- Check if the model contains hourglass for pose estimation
  self.hg = opt.hg
end

-- Get image path
function PennCropDataset:_impath(idx)
  return string.format('%04d/%06d.jpg',self.ind2sub[idx][1],self.ind2sub[idx][2])
end

-- Load image
function PennCropDataset:_loadImage(idx)
  return image.load(paths.concat(self.dir,self:_impath(idx)))
end

-- Get center and scale
function PennCropDataset:_getCenterScale(im)
  assert(im:size():size() == 3)
  local w = im:size(3)
  local h = im:size(2)
  local x = (w+1)/2
  local y = (h+1)/2
  local scale = math.max(w,h)/200
  -- Small adjustment so cropping is less likely to take feet out
  y = y + scale * 15
  scale = scale * 1.25
  return {torch.Tensor({x,y}), scale}
end

-- Get dataset size
function PennCropDataset:size()
  return self.ind2sub:size(1)
end

function PennCropDataset:get(idx, train)
  -- Load image
  local im = self:_loadImage(idx)

  -- Get center and scale
  local center, scale = unpack(self:_getCenterScale(im))

  -- Transform image
  local im = img.crop(im, center, scale, 0, self.inputRes)

  -- Get projection
  local pts = self.part[idx]
  local proj = torch.zeros(pts:size())
  for i = 1, pts:size(1) do
    if pts[i][1] ~= 0 and pts[i][2] ~= 0 then
      proj[i] = img.transform(pts[i], center, scale, 0, self.outputRes, false, false)
    end
  end

  -- Generate heatmap
  local hm = torch.zeros(pts:size(1), self.outputRes, self.outputRes)
  for i = 1, pts:size(1) do
    if pts[i][1] ~= 0 and pts[i][2] ~= 0 then
      img.drawGaussian(hm[i], torch.round(proj[i]), 2)
    end
  end

  -- Load gt points
  gtpts = pts

  -- Augment data
  if self.hg then
    if train then
      local sFactor = 0.25
      local rFactor = 30
      local s = torch.randn(1):mul(sFactor):add(1):clamp(1-sFactor,1+sFactor)[1]
      local r = torch.randn(1):mul(rFactor):clamp(-2*rFactor,2*rFactor)[1]
      -- Color
      im[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      im[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      im[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      -- Scale & rotation
      if torch.uniform() <= .6 then r = 0 end
      local inp, out = self.inputRes, self.outputRes
      im = img.crop(im, {(inp+1)/2,(inp+1)/2}, inp*s/200, r, inp)
      hm = img.crop(hm, {(out+1)/2,(out+1)/2}, out*s/200, r, out)
      for i = 1, pts:size(1) do
        if pts[i][1] ~= 0 and pts[i][2] ~= 0 then
          proj[i] = img.transform(proj[i], {(out+1)/2,(out+1)/2}, out*s/200, r, out, false, false)
        end
      end
      -- Flip
      if torch.uniform() <= .5 then
        im = img.flip(im)
        hm = img.flip(img.shuffleLR(hm,'penn-crop'))
        proj = geometry.shuffleLR(proj,'penn-crop')
        local ind = proj:eq(0)
        proj[{{},1}] = self.outputRes - proj[{{},1}] + 1
        proj[ind] = 0
      end
    else
      -- Add flipped image
      local im_ = torch.zeros(2, unpack(im:size():totable()))
      im_[1] = im:clone()
      im_[2] = img.flip(im)
      im = im_
    end
  end

  -- Set input
  if self.hg then
    input = im
  else
    input = hm
  end

  return {
    input = input,
    repos = torch.zeros(pts:size(1),3),
    trans = torch.zeros(3),
    focal = torch.zeros(1),
    hmap = hm,
    proj = proj,
    gtpts = gtpts,
    center = center,
    scale = scale,
  }
end

return M.PennCropDataset