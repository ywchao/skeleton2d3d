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

function M.createModel(numPt, inputRes)
  local inp = nn.Identity()()

  -- Initial processing of the image
  local in1 = cudnn.SpatialConvolution(numPt,64,1,1,1,1,0,0)(inp)
  local in2 = cudnn.SpatialBatchNormalization(64)(in1)
  local in3 = cudnn.ReLU(true)(in2)

  local cntr = hourglass(4,64,in3)
  local view = nn.View(-1):setNumInputDims(3)(cntr)

  local dfc = (inputRes/2^4)^2*64

  -- Relative joint position
  local fc1 = nn.Linear(dfc,dfc/4)(view)
  local relu1 = cudnn.ReLU(true)(fc1)
  local repos = nn.Linear(dfc/4,numPt*3)(relu1)
  local repos = nn.View(-1,numPt,3)(repos)

  -- Final model
  local model = nn.gModule({inp}, {repos})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

return M