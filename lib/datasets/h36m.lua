
local matio = require 'matio'
local geometry = require 'lib/util/geometry'
local img = require 'lib/util/img'
local util = require 'common/util'

matio.use_lua_strings = true

local M = {}
local H36MDataset = torch.class('skeleton2d3d.H36MDataset', M)

function H36MDataset:__init(opt, split)
  self.split = split
  self.dir = paths.concat(opt.data, 'frames')
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  self.anno_file = paths.concat(opt.data, split .. '.mat')
  self.inputRes = opt.inputResHG
  self.outputRes = opt.inputRes
  -- Load annotation
  self.ind2sub = matio.load(self.anno_file, 'ind2sub')
  self.coord_w = matio.load(self.anno_file, 'coord_w')
  self.coord_c = matio.load(self.anno_file, 'coord_c')
  self.coord_p = matio.load(self.anno_file, 'coord_p')
  self.focal = matio.load(self.anno_file, 'focal')
  -- Convert to Penn Action's format
  if opt.penn then
    self.jointType = 'penn-crop'
    self.coord_w = self.coord_w:index(2,self:_getPennInd(3))
    self.coord_c = self.coord_c:index(3,self:_getPennInd(3))
    self.coord_p = self.coord_p:index(3,self:_getPennInd(2))
  else
    self.jointType = 'h36m'
  end
  -- Get number of joints
  self.numPt = self.coord_w:size(2) / 3
  -- Check if the model contains hourglass for pose estimation
  self.hg = opt.hg
  -- -- Get mean limb length
  -- self.mean = self:_getMeanLimbLen(opt.penn)
  -- Remove corrupted images
  if self.hg then
    local rm = {{11,2,2}}
    for i = 1, #rm do
      local i1 = self.ind2sub[{{},1}]:eq(rm[i][1])
      local i2 = self.ind2sub[{{},2}]:eq(rm[i][2])
      local i3 = self.ind2sub[{{},3}]:eq(rm[i][3])
      local keep = util.find((i1+i2+i3):ne(3),1)
      self.ind2sub = self.ind2sub:index(1,keep)
      self.coord_w = self.coord_w:index(1,keep)
      self.coord_c = self.coord_c:index(2,keep)
      self.coord_p = self.coord_p:index(2,keep)
    end
  end
end

-- Get Penn Action's joint indices
function H36MDataset:_getPennInd(dim)
  local joints = {10,15,12,16,13,17,14,2,5,3,6,4,7}
  local ind = {}
  for _, v in ipairs(joints) do
    for i = 1, dim do
      table.insert(ind, (v-1)*dim+i)
    end
  end
  return torch.LongTensor(ind)
end

-- -- Get mean limb length
-- function H36MDataset:_getMeanLimbLen(penn)
--   local conn
--   if penn then
--     conn = torch.LongTensor{
--         { 2, 1}, { 3, 1}, { 4, 2}, { 5, 3}, { 6, 4}, { 7, 5},
--         { 8, 2}, { 9, 3}, {10, 8}, {11, 9}, {12,10}, {13,11}
--     }
--   else
--     conn = torch.LongTensor{
--         { 1, 2}, { 2, 3}, { 3, 4}, { 1, 5}, { 5, 6}, { 6, 7},
--         { 1, 8}, { 8, 9}, { 9,10}, {10,11},
--         { 9,12}, {12,13}, {13,14}, { 9,15}, {15,16}, {16,17}
--     }
--   end
--   local coord_w = self.coord_w:contiguous():view(-1,self.coord_w:size(2)/3,3)
--   local d1 = coord_w:index(2,conn:select(2,1))
--   local d2 = coord_w:index(2,conn:select(2,2))
--   return torch.csub(d1,d2):pow(2):sum(3):sqrt():mean(1):squeeze()
-- end

-- Load 3d pose in world coordinates
function H36MDataset:_loadPoseWorld(idx)
  return self.coord_w[idx]:contiguous():view(-1,3):double()
end

-- Load 3d pose in camera coordinates
function H36MDataset:_loadPoseCamera(idx, cam)
  return self.coord_c[cam][idx]:contiguous():view(-1,3):double()
end

-- Load 2d pose projection
function H36MDataset:_loadPoseProject(idx, cam)
  return self.coord_p[cam][idx]:contiguous():view(-1,2):double()
end

-- Load focal length
function H36MDataset:_loadFocal(cam)
  return self.focal[cam]
end

-- Normalize 3d pose to zero mean
function H36MDataset:_normalizePose(pose)
  local cntr = pose:mean(1)
  local pose = pose - cntr:expand(pose:size())
  return pose, cntr:view(3)
end

-- Sample projection
function H36MDataset:_sampleProj(pose_w)
  local res = torch.Tensor{{self.outputRes,self.outputRes}}
  local c = torch.div(res,2)

  local trial = 0
  local cam, proj, depth, pose_c
  while true do
    trial = trial + 1
    -- Sample focal length
    local mu = 1150 * self.outputRes / 1000
    local std = 450 * self.outputRes / 1000
    local f = torch.randn(1,1):mul(std):add(mu):expand(1,2)
    -- Sample translation
    local r = torch.rand(1)[1] * 4000 + 1000                       -- 1000 to 5000
    local az = (torch.rand(1)[1] - 0.5000) * 360 * math.pi / 180;  -- -180 to +180
    local el = (torch.rand(1)[1] - 0.1429) *  35 * math.pi / 180;  --   -5 to  +30
    local x = r * torch.cos(el) * torch.cos(az)
    local y = r * torch.cos(el) * torch.sin(az)
    local z = r * torch.sin(el)
    local T = torch.Tensor{{x,y,z}}
    -- Sample rotation
    local r1 = (torch.rand(1)[1] - 0.5000) *  10 * math.pi / 180;  --   -5 to   +5
    local r2 = (torch.rand(1)[1] - 0.5000) *  90 * math.pi / 180;  --  -45 to  +45
    local r3 = (torch.rand(1)[1] - 0.8571) *  35 * math.pi / 180;  --  -30 to   +5
    local R = geometry.angle2dcm(r1,r2,r3,'zyx') *
              geometry.angle2dcm(math.pi/2,0,-math.pi/2,'zyx') *
              geometry.angle2dcm(az,-el,0,'zyx')
    -- Get projection, depth, and pose in camera coordinates
    proj, depth, pose_c = geometry.camProject(pose_w, R, T, f, c)
    -- skip if any point is behind the image plane
    if depth:lt(f[1][1]):sum() > 0 then
      goto continue
    end
    -- skip if number of visible joints is below threshold
    local joint_thres = 0.999
    local c1 = torch.cmul(proj[{{},1}]:ge(1),proj[{{},1}]:le(res[1][1]))
    local c2 = torch.cmul(proj[{{},2}]:ge(1),proj[{{},2}]:le(res[1][2]))
    if torch.cmul(c1,c2):sum() < proj:size(1) * joint_thres then
      goto continue
    end
    -- skip if human is too small
    local area_thres = 0.1
    local x1 = proj[{{},1}]:min()
    local y1 = proj[{{},2}]:min()
    local x2 = proj[{{},1}]:max()
    local y2 = proj[{{},2}]:max()
    local area = (x2-x1)*(y2-y1)
    if area < res:prod() * area_thres then
      goto continue
    end
    -- Break
    cam = {res = res, c = c, T = T, R = R, f = f}
    break
    -- Contiunue
    ::continue::
  end
  return pose_c, cam.f, proj, depth, trial
end

-- Get image path
function H36MDataset:_impath(idx, cam)
  return string.format('%02d/%02d/%1d/%1d_%04d.jpg',
      self.ind2sub[idx][1],self.ind2sub[idx][2],
      self.ind2sub[idx][3],cam,self.ind2sub[idx][4])
end

-- Load image
function H36MDataset:_loadImage(idx, cam)
  return image.load(paths.concat(self.dir,self:_impath(idx,cam)))
end

-- Get center and scale
function H36MDataset:_getCenterScale(im)
  assert(im:size():size() == 3)
  local w = im:size(3)
  local h = im:size(2)
  local x = (w+1)/2
  local y = (h+1)/2
  local scale = ((w+h)/2)/200
  -- Small adjustment so cropping is less likely to take feet out
  -- y = y + scale * 15
  -- scale = scale * 1.25
  return {torch.Tensor({x,y}), scale}
end

-- Get dataset size
function H36MDataset:size()
  return self.ind2sub:size(1)
end

function H36MDataset:get(idx, train)
  local im, repos, trans, focal, proj
  local pose_w, pose_h, pose_c
  if self.hg then
    local cam
    if train then
      cam = torch.random(1,4)
    else
      cam = (idx - 1) % 4 + 1
    end
    im = self:_loadImage(idx, cam)
    -- Get resizing factor
    local factor = self.outputRes / ((im:size(2)+im:size(3))/2)
    focal = self:_loadFocal(cam):mean(1) * factor
    -- Transform image
    local center, scale = unpack(self:_getCenterScale(im))
    im = img.crop(im, center, scale, 0, self.inputRes)
    -- Load pose
    pose_c = self:_loadPoseCamera(idx, cam)
    repos, trans = self:_normalizePose(pose_c)
    -- Load and transform projection
    proj = self:_loadPoseProject(idx, cam)
    for i = 1, proj:size(1) do
      proj[i] = img.transform(proj[i], center, scale, 0, self.outputRes, false, false)
    end
  else
    pose_w = self:_loadPoseWorld(idx)
    pose_h = self:_normalizePose(pose_w)
    pose_c, focal, proj, _, _ = self:_sampleProj(pose_h)
    repos, trans = self:_normalizePose(pose_c)
    focal = focal[{{},1}]
  end

  local hm = torch.zeros(proj:size(1), self.outputRes, self.outputRes)
  for i = 1, proj:size(1) do
    img.drawGaussian(hm[i], proj[i], 2)
  end

  -- Augment data
  if self.hg then
    if train then
      -- Color
      im[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      im[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      im[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      -- Flip
      if torch.uniform() <= .5 then
        im = img.flip(im)
        hm = img.flip(img.shuffleLR(hm,self.jointType))
        repos = geometry.flip(geometry.shuffleLR(repos,self.jointType))
        trans = geometry.flip(trans:view(1,3)):view(3)
        proj = geometry.shuffleLR(proj,self.jointType)
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
    repos = repos,
    trans = trans,
    focal = focal,
    hmap = hm,
    proj = proj,
    mean = torch.zeros(proj:size(1)),
    -- mean = self.mean,
  }
end

return M.H36MDataset