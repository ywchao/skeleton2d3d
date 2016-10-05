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
  local low3, cntr

  if n > 1 then low3, cntr = hourglass(n-1,f,low2)
  else
    low3 = Residual(f,f)(low2)
    cntr = low3
  end

  local low4 = Residual(f,f)(low3)
  local up2 = nn.SpatialUpSamplingNearest(2)(low4)

  -- Bring two branches together
  return nn.CAddTable()({up1,up2}), cntr
end

local function lin(numIn, numOut, inp)
  -- Apply 1x1 convolution, no stride, no padding
  local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
  return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function M.createModel(numPt)
  local inp = nn.Identity()()

  -- Initial processing of the image
  local in1 = cudnn.SpatialConvolution(numPt,32,1,1,1,1,0,0)(inp)
  local in2 = cudnn.SpatialBatchNormalization(32)(in1)
  local in3 = cudnn.ReLU(true)(in2)

  -- Hourglass
  local hg, cntr = hourglass(4,32,in3)

  -- Linear layers to produce first set of predictions
  local ll = lin(32,32,hg)

  -- Output depth map
  local dm = cudnn.SpatialConvolution(32,numPt,1,1,1,1,0,0)(ll)

  -- Output focal length
  local view = nn.View(-1):setNumInputDims(3)(cntr)
  local fc1 = nn.Linear(512,128)(view)
  local relu1 = cudnn.ReLU(true)(fc1)
  local fc2 = nn.Linear(128,32)(relu1)
  local relu2 = cudnn.ReLU(true)(fc2)
  local fl = nn.Linear(32,1)(relu2)

  -- Final model
  local model = nn.gModule({inp}, {dm, fl})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

return M