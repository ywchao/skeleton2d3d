require 'nngraph'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
  -- Get model
  local model
  if checkpoint then
    local modelPath = paths.concat(opt.save, checkpoint.modelFile)
    assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
    print('=> Resuming model from ' .. modelPath)
    model = torch.load(modelPath)
  elseif opt.modelPath ~= 'none' then
    print('=> Loading trained model from' .. opt.modelPath)
    model = torch.load(opt.modelPath)
  else
    print('=> Creating model from file: lib/models/' .. opt.netType .. '.lua')
    local Model = require('lib/models/' .. opt.netType)
  
    -- Get output dim
    local Dataset = require('lib/datasets/' .. opt.dataset)
    local dataset = Dataset(opt, 'train')
    local numPt = dataset.coord_w:size(2) / 3

    -- Create model
    model = Model.createModel(numPt, opt.inputRes)
  end

  -- Create criterion
  local criterion
  if #model.outnode.children == 1 then
    criterion = nn.MSECriterion()
  else
    criterion = nn.ParallelCriterion()
    for i = 1, #model.outnode.children do
      criterion:add(nn.MSECriterion())
    end
    if #model.outnode.children == 3 then
      criterion.weights[1] = 1
      criterion.weights[2] = opt.weightTrans
      criterion.weights[3] = opt.weightFocal
    end
    if #model.outnode.children == 4 then
      criterion.weights[1] = 1
      criterion.weights[2] = opt.weightTrans
      criterion.weights[3] = opt.weightFocal
      criterion.weights[4] = opt.weightProj
    end
  end

  -- Convert to CUDA
  model:cuda()
  criterion:cuda()

  return model, criterion
end

return M