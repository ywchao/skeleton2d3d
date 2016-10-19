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
    local numPt = dataset.numPt

    -- Create model
    model = Model.createModel(numPt, opt.inputRes)

    -- Load trained models
    if opt.hg then
      if opt.hgModel ~= 'none' then
        assert(paths.filep(opt.hgModel),
            'initial hourglass model not found: ' .. opt.hgModel)
        local model_hg = torch.load(opt.hgModel)
        Model.loadHourglass(model, model_hg, opt.hgFix)
      end
      if opt.s3Model ~= 'none' then
        assert(paths.filep(opt.s3Model),
            'initial skel3dnet model not found: ' .. opt.s3Model)
        local model_s3 = torch.load(opt.s3Model)
        Model.loadSkel3DNet(model, model_s3, opt.s3Fix)
      end
    end
  end

  -- Create criterion
  local criterion
  local nOutput = #model.outnode.children
  if nOutput == 1 then
    criterion = nn.MSECriterion()
  else
    criterion = nn.ParallelCriterion()
    for i = 1, nOutput do
      criterion:add(nn.MSECriterion())
    end
  end
  if opt.hg then
    assert(nOutput == 5 or nOutput == 6)
    criterion.weights[2] = 0
    criterion.weights[3] = 0
    criterion.weights[4] = 0
    if nOutput == 5 then
      criterion.weights[1] = 1
      criterion.weights[5] = opt.weightProj
    end
    if nOutput == 6 then
      criterion.weights[1] = 1
      criterion.weights[5] = opt.weightProj
      criterion.weights[6] = opt.weightLLPrior
    end
  else
    assert(nOutput == 1 or nOutput == 3 or nOutput == 4)
    if nOutput == 3 then
      criterion.weights[1] = 1
      criterion.weights[2] = opt.weightTrans
      criterion.weights[3] = opt.weightFocal
    end
    if nOutput == 4 then
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