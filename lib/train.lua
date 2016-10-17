require 'cunn'
require 'optim'

local matio = require 'matio'
local Logger = require 'lib/util/Logger'
local geometry = require 'lib/util/geometry'
local img = require 'lib/util/img'
local eval = require 'lib/util/eval'
local util = require 'common/util'

local M = {}
local Trainer = torch.class('skeleton2d3d.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  self.model = model
  self.criterion = criterion
  self.optimState = optimState or {
    learningRate = opt.LR,
    weightDecay = opt.weightDecay,
  }
  self.opt = opt
  self.params, self.gradParams = model:getParameters()
  self.logger = {
    train = Logger(paths.concat(opt.save, 'train.log'), opt.resume),
    val = Logger(paths.concat(opt.save, 'val.log'), opt.resume),
  }
  self:initLogger(self.logger['train'])
  self:initLogger(self.logger['val'])
  -- Set log format
  if self.opt.hg then
    self.format_l = '%7.4f'
    self.format_e = '%7.4f'
  else
    self.format_l = '%7.0f'
    self.format_e = '%7.2f'
  end
  self.nOutput = #self.model.outnode.children
  self.jointType = opt.dataset
  if opt.penn then
    self.jointType = 'penn-crop'
  end
end

function Trainer:initLogger(logger)
  local names = {}
  names[1] = 'epoch'
  names[2] = 'iter'
  names[3] = 'time'
  names[4] = 'datTime'
  names[5] = 'loss'
  names[6] = 'err'
  names[7] = 'acc'
  logger:setNames(names)
end

function Trainer:train(epoch, loaders)
  local timer = torch.Timer()
  local dataTimer = torch.Timer()

  local function feval()
    return self.criterion.output, self.gradParams
  end

  local dataloader = loaders['train']
  local size = dataloader:size()

  print(('=> Training epoch # %d'):format(epoch))
  xlua.progress(0, size)

  -- Set the batch norm to training mode
  self.model:training()
  for i, sample in dataloader:run({train=true}) do
    local dataTime = dataTimer:time().real
  
    -- Get input/output and convert to CUDA
    local input = sample.input:cuda()
    local repos = sample.repos:cuda()
    local trans = sample.trans:cuda()
    local focal = sample.focal:cuda()
    local hmap = sample.hmap:cuda()
    local proj = sample.proj:cuda()
    local mean = sample.mean:cuda()
    
    -- Get target
    local target
    if not self.opt.hg then
      if self.nOutput == 1 then target = repos end
      if self.nOutput == 3 then target = {repos, trans, focal} end
      if self.nOutput == 4 then target = {repos, trans, focal, proj} end
    end

    -- Forward pass
    local output = self.model:forward(input)
    if self.opt.hg then
      local proj_ = proj:clone()
      proj_[proj_:eq(0)] = output[2][proj_:eq(0)]
      if self.nOutput == 2 then target = {hmap, proj_} end
      if self.nOutput == 3 then target = {hmap, proj_, mean} end
    end
    local loss = self.criterion:forward(self.model.output, target)

    -- Backprop
    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, target)
    self.model:backward(input, self.criterion.gradInput)

    -- Optimization
    optim.rmsprop(feval, self.params, self.optimState)

    -- Compute mean per joint position error (MPJPE)
    if self.nOutput == 1 then output = {output} end
    local pred, err, acc
    if self.opt.hg then
      proj = proj:float()
      if self.opt.hgFix then pred = output[2]:float() end
      if self.opt.s3Fix then pred = eval.getPreds(output[1]:float()) end
      if self.opt.dataset == 'penn-crop' then
        err = self:_computeError(proj,pred)
      else
        err = torch.csub(proj,pred):pow(2):sum(3):sqrt():mean(2):mean()
      end
      acc = self:_computeAccuracy(proj,pred)
    else
      repos = repos:float()
      pred = output[1]:float()
      err = torch.csub(repos,pred):pow(2):sum(3):sqrt():mean(2):mean()
      acc = 0/0
    end

    -- Print and log
    local time = timer:time().real
    local entry = {}
    entry[1] = string.format("%d" % epoch)
    entry[2] = string.format("%d" % i)
    entry[3] = string.format("%.3f" % time)
    entry[4] = string.format("%.3f" % dataTime)
    entry[5] = string.format(self.format_l % loss)
    entry[6] = string.format(self.format_e % err)
    entry[7] = string.format("%7.5f" % acc)
    self.logger['train']:add(entry)
  
    xlua.progress(i, size)

    timer:reset()
    dataTimer:reset()
  end
end

function Trainer:test(epoch, iter, loaders, split)
  local testTimer = torch.Timer()

  local dataloader = loaders[split]
  local size = dataloader:sizeDataset()
  local lossSum, errSum, accSum, N = 0.0, 0.0, 0.0, 0.0

  print("=> Test on " .. split)
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run({train=false}) do
    -- Get input/output and convert to CUDA
    local input = sample.input:cuda()
    local repos = sample.repos:cuda()
    local trans = sample.trans:cuda()
    local focal = sample.focal:cuda()
    local hmap = sample.hmap:cuda()
    local proj = sample.proj:cuda()
    local mean = sample.mean:cuda()

    -- Get target
    local target
    if not self.opt.hg then
      if self.nOutput == 1 then target = repos end
      if self.nOutput == 3 then target = {repos, trans, focal} end
      if self.nOutput == 4 then target = {repos, trans, focal, proj} end
    end

    -- Forward pass
    local output = self.model:forward(input)
    if self.opt.hg then
      local hmap1 = output[1][{{1}}]
      local hmap2 = img.flip(img.shuffleLR(output[1][{{2}}],self.jointType))
      local proj1 = output[2][{{1}}]
      local proj2 = output[2][{{2}}]
      proj2 = proj2 - (self.opt.inputRes+1)/2
      proj2 = geometry.flip(geometry.shuffleLR(proj2,self.jointType))
      proj2 = proj2 + (self.opt.inputRes+1)/2
      output[1] = torch.add(hmap1,hmap2):div(2)
      output[2] = torch.add(proj1,proj2):div(2)
      if self.nOutput == 3 then output[3] = output[3]:mean(1) end
      local proj_ = proj:clone()
      proj_[proj_:eq(0)] = output[2][proj_:eq(0)]
      if self.nOutput == 2 then target = {hmap, proj_} end
      if self.nOutput == 3 then target = {hmap, proj_, mean} end
    end
    local loss = self.criterion:forward(output, target)

    -- Compute mean per joint position error (MPJPE)
    if self.opt.hg then
      assert(input:size(1) == 2, 'batch size must be 2 with run({train=false})')
    else
      assert(input:size(1) == 1, 'batch size must be 1 with run({train=false})')
    end
    if self.nOutput == 1 then output = {output} end
    local pred, err, acc
    if self.opt.hg then
      proj = proj:float()
      if self.opt.hgFix then pred = output[2]:float() end
      if self.opt.s3Fix then pred = eval.getPreds(output[1]:float()) end
      if self.opt.dataset == 'penn-crop' then
        err = self:_computeError(proj,pred)
      else
        err = torch.csub(proj,pred):pow(2):sum(3):sqrt():mean()
      end
      acc = self:_computeAccuracy(proj,pred)
    else
      if self.opt.dataset == 'penn-crop' then
        assert(self.nOutput == 4)
        proj = proj:float()
        pred = output[4]:float()
        err = self:_computeError(proj,pred)
        acc = self:_computeAccuracy(proj,pred)
      else
        repos = repos:float()
        pred = output[1]:float()
        err = torch.csub(repos,pred):pow(2):sum(3):sqrt():mean()
        acc = 0/0
      end
    end

    lossSum = lossSum + loss
    errSum = errSum + err
    accSum = accSum + acc
    N = N + 1

    xlua.progress(i, size)
  end
  self.model:training()

  local loss = lossSum / N
  local err = errSum / N
  local acc = accSum / N

  -- Print and log
  local testTime = testTimer:time().real
  local entry = {}
  entry[1] = string.format("%d" % epoch)
  entry[2] = string.format("%d" % iter)
  entry[3] = string.format("%.3f" % testTime)
  entry[4] = string.format("%.3f" % 0/0)
  entry[5] = string.format(self.format_l % loss)
  entry[6] = string.format(self.format_e % err)
  entry[7] = string.format("%7.5f" % acc)
  self.logger['val']:add(entry)

  return err, acc
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local size = dataloader:sizeDataset()
  local inds = torch.IntTensor(size)
  local poses, repos, trans, focal, proj

  print("=> Generating predictions ...")
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run({train=false}) do
    -- Get input and convert to CUDA
    local index = sample.index
    local input = sample.input:cuda()

    -- Forward pass
    local output = self.model:forward(input)
    if self.opt.hg then
      local hmap1 = output[1][{{1}}]
      local hmap2 = img.flip(img.shuffleLR(output[1][{{2}}],self.jointType))
      local proj1 = output[2][{{1}}]
      local proj2 = output[2][{{2}}]
      proj2 = proj2 - (self.opt.inputRes+1)/2
      proj2 = geometry.flip(geometry.shuffleLR(proj2,self.jointType))
      proj2 = proj2 + (self.opt.inputRes+1)/2
      output[1] = torch.add(hmap1,hmap2):div(2)
      output[2] = torch.add(proj1,proj2):div(2)
      if self.nOutput == 3 then output[3] = output[3]:mean(1) end
    end

    -- Copy output
    if self.opt.hg then
      assert(input:size(1) == 2, 'batch size must be 2 with run({train=false})')
    else
      assert(input:size(1) == 1, 'batch size must be 1 with run({train=false})')
    end
    inds[i] = index[1]
    if self.nOutput == 1 then output = {output} end

    if self.opt.hg then
      local repos_p = self.model.modules[self.model.id_repos].output:float()[1]
      local trans_p = self.model.modules[self.model.id_trans].output:float()[1]
      local focal_p = self.model.modules[self.model.id_focal].output:float()[1]
      if not poses then
        poses = torch.FloatTensor(size, unpack(sample.proj[1]:size():totable()))
      end
      if not repos then
        repos = torch.FloatTensor(size, unpack(repos_p:size():totable()))
      end
      if not trans then
        trans = torch.FloatTensor(size, unpack(trans_p:size():totable()))
      end
      if not focal then
        focal = torch.FloatTensor(size, unpack(focal_p:size():totable()))
      end
      if not proj then
        proj = torch.FloatTensor(size, unpack(output[2][1]:size():totable()))
      end
      poses[i]:copy(sample.proj[1])
      repos[i]:copy(repos_p)
      trans[i]:copy(trans_p)
      focal[i]:copy(focal_p)
      proj[i]:copy(output[2]:float()[1])

      -- Save heatmap output
      local hmap_path = paths.concat(self.opt.save,'hmap_' .. split)
      local hmap_file = paths.concat(hmap_path, string.format("%05d.mat" % index[1]))
      util.makedir(hmap_path)
      matio.save(hmap_file, {hmap = output[1]:float()[1]})
    else
      if not poses then
        poses = torch.FloatTensor(size, unpack(sample.repos[1]:size():totable()))
      end
      if not repos then
        repos = torch.FloatTensor(size, unpack(output[1][1]:size():totable()))
      end
      poses[i]:copy(sample.repos[1] + sample.trans[1]:view(1,3):expand(sample.repos[1]:size()))
      repos[i]:copy(output[1]:float()[1])

      if self.nOutput > 1 then
        if not trans then
          trans = torch.FloatTensor(size, unpack(output[2][1]:size():totable()))
        end
        if not focal then
          focal = torch.FloatTensor(size, unpack(output[3][1]:size():totable()))
        end
        trans[i]:copy(output[2]:float()[1])
        focal[i]:copy(output[3]:float()[1])
      end
      if self.nOutput > 3 then
        if not proj then
          proj = torch.FloatTensor(size, unpack(output[4][1]:size():totable()))
        end
        proj[i]:copy(output[4]:float()[1])
      end
    end

    xlua.progress(i, size)
  end
  self.model:training()

  -- Sort preds by inds
  local inds, i = torch.sort(inds)
  poses = poses:index(1, i)
  repos = repos:index(1, i)
  if trans then trans = trans:index(1, i) end
  if focal then focal = focal:index(1, i) end
  if proj then proj = proj:index(1, i) end
  
  -- Save final predictions
  matio.save(self.opt.save .. '/preds_' .. split .. '.mat',
      {poses = poses, repos = repos, trans = trans, focal = focal, proj = proj})
end

function Trainer:_computeError(target, output)
-- target: N x d x 2
-- output: N x d x 2
  local e, n = {}, {}
  for i = 1, target:size(1) do
    e[i], n[i] = 0.0, 0.0
    for j = 1, target:size(2) do
      if target[i][j][1] > 0 then
        local p1 = target:select(2,j)
        local p2 = output:select(2,j)
        n[i] = n[i] + 1
        e[i] = e[i] + torch.csub(p1,p2):pow(2):sum(2):sqrt()[1][1]
      end
    end
  end
  return torch.cdiv(torch.Tensor(e),torch.Tensor(n)):mean()
end

function Trainer:_computeAccuracy(target, output)
-- target: N x d x 2
-- output: N x d x 2
  local jntIdxs
  if self.opt.jointType == 'penn-crop' then
    jntIdxs = {4,5,6,7,8,9,10,11,12,13}
  end
  return eval.coordAccuracy(output,target,nil,jntIdxs,self.opt.inputRes)
end

return M.Trainer