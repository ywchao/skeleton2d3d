require 'hdf5'
require 'image'

local matio = require 'matio'
local img = require 'lib/util/img'

local M = {}
local PennCropDataset = torch.class('skeleton2d3d.PennCropDataset', M)

function PennCropDataset:__init(opt, split)
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  -- Load annotation
  annot_file = paths.concat(opt.data, split .. '.h5')
  self.ind2sub = hdf5.open(annot_file,'r'):read('ind2sub'):all()
  self.visible = hdf5.open(annot_file,'r'):read('visible'):all()
  self.part = hdf5.open(annot_file,'r'):read('part'):all()
  -- Get input and output resolution
  self.inputRes = opt.inputResHG
  self.outputRes = opt.inputRes
  -- Get number of joints
  self.numPt = self.visible:size(2)
  -- Check if the model contains hourglass for pose estimation
  self.hg = opt.hg
  -- Get mean limb length
  self.mean = self:_getMeanLimbLen()
end

-- Get mean limb length
function PennCropDataset:_getMeanLimbLen()
  local joints = torch.LongTensor{10,15,12,16,13,17,14,2,5,3,6,4,7}
  local conn = torch.LongTensor{
      { 2, 1}, { 3, 1}, { 4, 2}, { 5, 3}, { 6, 4}, { 7, 5},
      { 8, 2}, { 9, 3}, {10, 8}, {11, 9}, {12,10}, {13,11}
  }
  local coord_w = matio.load('./data/h36m/train.mat','coord_w')
  coord_w = coord_w:contiguous():view(-1,coord_w:size(2)/3,3)
  coord_w = coord_w:index(2,joints)
  local d1 = coord_w:index(2,conn:select(2,1))
  local d2 = coord_w:index(2,conn:select(2,2))
  return torch.csub(d1,d2):pow(2):sum(3):sqrt():mean(1):squeeze()
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
  -- local w = im:size(2)
  -- local h = im:size(3)
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
  local vis = self.visible[idx]
  local proj = torch.zeros(pts:size())
  for i = 1, pts:size(1) do
    if vis[i] == 1 then
      proj[i] = img.transform(pts[i], center, scale, 0, self.outputRes, false, false)
      -- proj[i] = img.transform(torch.add(pts[i],1), center, scale, 0, self.outputRes, false)
    end
  end

  -- Generate heatmap
  local hm = torch.zeros(pts:size(1), self.outputRes, self.outputRes)
  for i = 1, pts:size(1) do
    if vis[i] == 1 then
      img.drawGaussian(hm[i], torch.round(proj[i]), 2)
      -- img.drawGaussian(hm[i], proj[i]:int(), 2)
      -- img.drawGaussian(hm[i], proj[i], 2)
    end
  end

  -- -- Artificially create proj for fair comparison with pose-hg-train
  -- eval = require 'lib/util/eval'
  -- proj = eval.getPreds(hm:view(1,unpack(hm:size():totable())))
  -- -- proj = proj:permute(2,3,1):squeeze()
  -- proj = proj:view(pts:size())

  -- Set input
  if self.hg then
    input = im
  else
    input = hm
  end

  -- Add flipped input for prediction
  if self.hg and not train then
    local input_ = torch.zeros(2, unpack(input:size():totable()))
    input_[1] = input:clone()
    input_[2] = img.flip(input)
    input = input_
  end

  return {
    input = input,
    repos = torch.zeros(1),
    trans = torch.zeros(1),
    focal = torch.zeros(1),
    proj = proj,
    mean = self.mean,
  }
end

return M.PennCropDataset