require 'nngraph'
require 'cudnn'

require 'lib/models/Residual'

local M = {}

local function hourglass(n, f, inp)
  -- Upper branch
  local up1 = Residual(f,f)(inp)

  -- Lower branch
  local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
  local low2 = Residual(f,f)(low1)
  local low3
  if n > 1 then low3 = hourglass(n-1,f,low2)
  else
      low3 = Residual(f,f)(low2)
  end
  local low4 = Residual(f,f)(low3)
  local up2 = nn.SpatialUpSamplingNearest(2)(low4)

  -- Bring two branches together
  return nn.CAddTable()({up1,up2})
end

local function lin(numIn, numOut, inp)
  -- Apply 1x1 convolution, stride 1, no padding
  local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

local function tieWeightBiasOneModule(module1, module2)
  if module2.modules ~= nil then
    assert(module1.modules)
    assert(#module1.modules == #module2.modules)
    for i = 1, #module1.modules do
      tieWeightBiasOneModule(module1.modules[i], module2.modules[i])
    end
  end
  if module2.weight ~= nil then
    assert(module1.weight)
    assert(module1.gradWeight)
    assert(module2.gradWeight)
    assert(module2.weight:numel() == module1.weight:numel())
    assert(module2.gradWeight:numel() == module1.gradWeight:numel())
    assert(torch.typename(module1) == torch.typename(module2))
    module2.weight = module1.weight
    module2.gradWeight = module1.gradWeight
  end
  if module2.bias ~= nil then
    assert(module1.bias)
    assert(module1.gradBias)
    assert(module2.gradBias)
    assert(module2.bias:numel() == module1.bias:numel())
    assert(module2.gradBias:numel() == module1.gradBias:numel())
    assert(torch.typename(module1) == torch.typename(module2))
    module2.bias = module1.bias
    module2.gradBias = module1.gradBias
  end
  if module2.running_mean ~= nil then
    assert(module1.running_mean)
    assert(module1.running_var)
    assert(module2.running_var)
    assert(module2.running_mean:numel() == module1.running_mean:numel())
    assert(module2.running_var:numel() == module1.running_var:numel())
    assert(torch.typename(module1) == torch.typename(module2))
    module2.running_mean = module1.running_mean
    module2.running_var = module1.running_var
  end
end

function M.createModel(numPt, outputRes)
  local inp = nn.Identity()()

  -- Initial processing of the image
  local in1 = cudnn.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)
  local in2 = nn.SpatialBatchNormalization(64)(in1)
  local in3 = cudnn.ReLU(true)(in2)
  local in4 = Residual(64,128)(in3)
  local in5 = cudnn.SpatialMaxPooling(2,2,2,2)(in4)
  local in6 = Residual(128,128)(in5)
  local in7 = Residual(128,256)(in6)

  -- Hourglass
  local hg = hourglass(4,256,in7)

  -- Linear layer to produce first set of predictions
  local ll = lin(256,256,hg)

  -- Predicted heatmaps
  local out = cudnn.SpatialConvolution(256,numPt,1,1,1,1,0,0)(ll)

  -- Final model
  local model = nn.gModule({inp}, {out})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

function M.loadHourglass(model, model_hg, fix)
  -- Load weight and bias
  for i = 1, #model_hg.modules do
    local name = torch.typename(model.modules[i])
    local name_hg = torch.typename(model_hg.modules[i])
    assert(name == name_hg, 'weight loading error: class name mismatch')
    tieWeightBiasOneModule(model_hg.modules[i], model.modules[i])
    if fix then
      fixWeightBiasOneModule(model.modules[i])
    end
  end

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()
end

return M