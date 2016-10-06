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
  local in1 = cudnn.SpatialConvolution(numPt,64,1,1,1,1,0,0)(inp)
  local in2 = cudnn.SpatialBatchNormalization(64)(in1)
  local in3 = cudnn.ReLU(true)(in2)

  local cntr = hourglass(4,64,in3)
  local view = nn.View(-1):setNumInputDims(3)(cntr)

  local fc = nn.Linear(1024,256)(view)
  local relu = cudnn.ReLU(true)(fc)
  local repos = nn.Linear(256,3*numPt)(relu)
  local repos = nn.View(-1,3,numPt)(repos)

  -- Final model
  local model = nn.gModule({inp}, {repos})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

return M