
local matio = require 'matio'
local geometry = require 'lib/util/geometry'
local img = require 'lib/util/img'

matio.use_lua_strings = true

local M = {}
local H36MDataset = torch.class('skeleton2d3d.H36MDataset', M)

function H36MDataset:__init(opt, split)
  self.split = split
  self.anno_file = paths.concat(opt.data, split .. '.mat')
  self.inputRes = opt.inputRes
  -- Load annotation
  self.ind2sub = matio.load(self.anno_file, 'ind2sub')
  self.coord_w = matio.load(self.anno_file, 'coord_w')
  self.coord_c = matio.load(self.anno_file, 'coord_c')
  self.coord_p = matio.load(self.anno_file, 'coord_p')
  self.focal = matio.load(self.anno_file, 'focal')
end

-- Load 3d pose in world coordinates
function H36MDataset:_loadPoseWorld(idx)
  return self.coord_w[idx]:contiguous():view(-1,3):t():double()
end

-- Load 3d pose in camera coordinates
function H36MDataset:_loadPoseCamera(idx)
  return self.coord_c[idx]:contiguous():view(-1,3):t():double()
end

-- Load 2d pose projection
function H36MDataset:_loadPoseProject(idx)
  return self.coord_p[idx]:contiguous():view(-1,2):t():double()
end

-- Load focal length
function H36MDataset:_loadFocal(idx)
  return self.focal[idx]
end

-- Normalize 3d pose to zero mean
function H36MDataset:_normalizePose(pose)
  return pose - pose:mean(2):expand(pose:size())
end

-- Sample projection
function H36MDataset:_sampleProj(pose_w)
  local res = torch.Tensor{{self.inputRes,self.inputRes}}
  local c = torch.div(res,2)

  local trial = 0
  local cam, proj, depth, pose_c
  while true do
    trial = trial + 1
    -- Sample focal length
    local mu = 1150 * self.inputRes / 1000
    local std = 450 * self.inputRes / 1000
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
    proj, depth, pose_c = geometry.camProject(pose_w:t(), R, T, f, c)
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
  proj = proj:t()
  depth = depth:t()
  return pose_c, cam.f, proj, depth, trial
end

-- Get dataset size
function H36MDataset:size()
  return self.ind2sub:size(1)
end

function H36MDataset:get(idx, train)
  local pose
  local focal, proj, depth
  -- if train then
    local pose_w = self:_loadPoseWorld(idx)
    local pose_w = self:_normalizePose(pose_w)
    pose, focal, proj, depth, _ = self:_sampleProj(pose_w)
    focal = focal[1][1]
  -- else
  --   pose = self:_loadPoseCamera(idx)
  --   proj = self:_loadPoseProject(idx):mul(self.inputRes / 1000)
  --   focal = self:_loadFocal(idx):mul(self.inputRes / 1000):mean()
  --   depth = pose[{{3}}]
  -- end

  local hm = torch.zeros(proj:size(2), self.inputRes, self.inputRes)
  for i = 1, proj:size(2) do
    img.drawGaussian(hm[i], proj[{{},i}], 2)
  end

  local dm = hm:clone()
  for i = 1, proj:size(2) do
    dm[i]:mul(depth[1][i])
  end

  return {
    input = hm,
    depth = dm,
    focal = focal,
    pose = pose,
  }
end

return M.H36MDataset