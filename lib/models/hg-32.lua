require 'nngraph'
require 'cudnn'

require 'lib/models/Residual'

local M = {}

local function hourglass(n, f, inp)
  -- Lower branch
  local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
  local low2 = Residual(f,f)(low1)
  local low3
  if n > 1 then low3 = hourglass(n-1,f,low2)
  else
    low3 = Residual(f,f)(low2)
  end
  return low3
end

function M.createModel(numPt)
  local inp = nn.Identity()()

  -- Initial processing of the image
  local in1 = cudnn.SpatialConvolution(numPt,32,1,1,1,1,0,0)(inp)
  local in2 = cudnn.SpatialBatchNormalization(32)(in1)
  local in3 = cudnn.ReLU(true)(in2)

  local cntr = hourglass(4,32,in3)
  local view = nn.View(-1):setNumInputDims(3)(cntr)

  -- Output depth
  local fc1d = nn.Linear(512,128)(view)
  local relu1d = cudnn.ReLU(true)(fc1d)
  local depth = nn.Linear(128,numPt)(relu1d)
  local depth = nn.AddConstant(2500)(depth)

  -- Output focal length
  local fc1f = nn.Linear(512,128)(view)
  local relu1f = cudnn.ReLU(true)(fc1f)
  local focal = nn.Linear(128,1)(relu1f)
  local focal = nn.AddConstant(73.6)(focal)

  -- Final model
  local model = nn.gModule({inp}, {depth, focal})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

return M