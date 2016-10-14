require 'hdf5'
require 'image'

local img = require 'lib/util/img'

local M = {}
local PennCropDataset = torch.class('skeleton2d3d.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  -- self.opt = opt
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  -- Load annotation
  annot_file = paths.concat(opt.data, split .. '.h5')
  self.ind2sub = hdf5.open(annot_file,'r'):read('ind2sub'):all()
  self.visible = hdf5.open(annot_file,'r'):read('visible'):all()
  self.part = hdf5.open(annot_file,'r'):read('part'):all()
  -- Get input and output resolution
  self.inputRes = opt.inputRes
  self.outputRes = opt.inputRes
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

function PennCropDataset:get(idx)
  -- Load image
  local im = self:_loadImage(idx)

  -- Get center and scale
  local center, scale = unpack(self:_getCenterScale(im))

  -- Transform image
  local input = img.crop(im, center, scale, 0, self.inputRes)

  -- Generate target
  local pts = self.part[idx]
  local vis = self.visible[idx]
  local target = torch.zeros(pts:size(1), self.outputRes, self.outputRes)
  for i = 1, pts:size(1) do
    if vis[i] == 1 then
      img.drawGaussian(target[i], img.transform(pts[i], center, scale, 0, self.outputRes), 2)
    end
  end

  return {
    input = target,
    repos = torch.zeros(1),
    trans = torch.zeros(1),
    focal = torch.zeros(1),
    proj = torch.zeros(1),
  }
end

return M.PennCropDataset
