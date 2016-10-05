require 'cunn'
require 'optim'

local matio = require 'matio'
local Logger = require 'lib/util/Logger'
local geometry = require 'lib/util/geometry'

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
end

function Trainer:initLogger(logger)
  local names = {}
  names[1] = 'epoch'
  names[2] = 'iter'
  names[3] = 'time'
  names[4] = 'datTime'
  names[5] = 'loss'
  names[6] = 'err'
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
  
    -- Get input and depth and focal and convert to CUDA
    local input = sample.input:cuda()
    local depth = sample.depth:cuda()
    local focal = sample.focal:cuda()
    
    -- Forward pass
    local output = self.model:forward(input)
    local loss = self.criterion:forward(self.model.output, {depth, focal})

    -- Compute mean per joint position error (MPJPE)
    local pred_depth = output[1]:float()
    local pred_focal = output[2]:float()
    local pose = sample.pose
    local errSum = 0.0
    for j = 1, input:size(1) do
      local pred = geometry.backProjectHMDM(input[j], pred_depth[j], pred_focal[j][1]):float()
      local err = torch.csub(pred,pose[j]):pow(2):sum(1):sqrt():mean()
      errSum = errSum + err
    end
    local err = errSum / size

    -- Backprop
    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, {depth, focal})
    self.model:backward(input, self.criterion.gradInput)

    -- Optimization
    optim.rmsprop(feval, self.params, self.optimState)

    -- Print and log
    local time = timer:time().real
    local entry = {}
    entry[1] = string.format("%d" % epoch)
    entry[2] = string.format("%d" % i)
    entry[3] = string.format("%.3f" % time)
    entry[4] = string.format("%.3f" % dataTime)
    entry[5] = string.format("%7.f" % loss)
    entry[6] = string.format("%.2f" % err)
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
  local lossSum, errSum = 0.0, 0.0

  print("=> Test on " .. split)
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run({train=false}) do
    -- Get input and depth and focal and convert to CUDA
    local input = sample.input:cuda()
    local depth = sample.depth:cuda()
    local focal = sample.focal:cuda()

    -- Forward pass
    local output = self.model:forward(input)
    local loss = self.criterion:forward(self.model.output, {depth, focal})

    -- Compute mean per joint position error (MPJPE)
    assert(input:size(1) == 1, 'batch size must be 1 with run({train=false})')
    local pred_depth = output[1]:float()[1]
    local pred_focal = output[2]:float()[1]
    local pose = sample.pose
    local pred = geometry.backProjectHMDM(input[1], pred_depth, pred_focal[1]):float()
    local err = torch.csub(pred,pose):pow(2):sum(1):sqrt():mean()
    
    lossSum = lossSum + loss
    errSum = errSum + err

    xlua.progress(i, size)
  end
  self.model:training()

  local loss = lossSum / size
  local err = errSum / size

  -- Print and log
  local testTime = testTimer:time().real
  local entry = {}
  entry[1] = string.format("%d" % epoch)
  entry[2] = string.format("%d" % iter)
  entry[3] = string.format("%.3f" % testTime)
  entry[4] = string.format("%.3f" % 0/0)
  entry[5] = string.format("%7.f" % loss)
  entry[6] = string.format("%.2f" % err)
  self.logger['val']:add(entry)

  return err
end

function Trainer:predict(loaders, split)
  local dataloader = loaders[split]
  local size = dataloader:sizeDataset()
  local inds = torch.IntTensor(size)
  local preds, poses

  print("=> Generating predictions ...")
  xlua.progress(0, size)

  self.model:evaluate()
  for i, sample in dataloader:run({train=false}) do
    -- Get input and convert to CUDA
    local index = sample.index
    local input = sample.input:cuda()

    -- Forward pass
    local output = self.model:forward(input)

    -- Copy output
    assert(input:size(1) == 1, 'batch size must be 1 with run({train=false})')
    inds[i] = index[1]
    local pred_depth = output[1]:float()[1]
    local pred_focal = output[2]:float()[1]
    local pose = sample.pose
    local pred = geometry.backProjectHMDM(input[1], pred_depth, pred_focal[1]):float()
    
    if not preds then
      preds = torch.FloatTensor(size, unpack(pred:size():totable()))
    end
    if not poses then
      poses = torch.FloatTensor(size, unpack(pose[1]:size():totable()))
    end
    preds[i]:copy(pred)
    poses[i]:copy(pose[1])

    xlua.progress(i, size)
  end
  self.model:training()

  -- Sort preds by inds
  local inds, i = torch.sort(inds)
  preds = preds:index(1, i)
  poses = poses:index(1, i)
  
  -- Save final predictions
  matio.save(self.opt.save .. '/preds_' .. split .. '.mat', {preds = preds, poses = poses})
end

return M.Trainer
