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
  local repos = nn.Linear(dfc/4,3*numPt)(relu1)
  local repos = nn.View(-1,3,numPt)(repos)

  -- Translation of skeleton center
  local fc2 = nn.Linear(dfc,dfc/4)(view)
  local relu2 = cudnn.ReLU(true)(fc2)
  local trans = nn.Linear(dfc/4,3)(relu2)
  local txy = nn.Narrow(2,1,2)(trans)
  local td = nn.Narrow(2,3,1)(trans)
  local td = nn.AddConstant(2500)(td)
  local trans = nn.JoinTable(2)({txy,td})
  
  -- Focal length
  local fc3 = nn.Linear(dfc,dfc/4)(view)
  local relu3 = cudnn.ReLU(true)(fc3)
  local focal = nn.Linear(dfc/4,1)(relu3)
  local focal = nn.AddConstant(73.6)(focal)

  -- 3D points in camera coordinates
  local rept = nn.Replicate(numPt,3)(trans)
  local p3d = nn.CAddTable()({repos,rept})

  -- Projection
  local xy = nn.Narrow(2,1,2)(p3d)
  local d = nn.Select(2,3)(p3d)
  local repd = nn.Replicate(2,2)(d)
  local proj = nn.CDivTable()({xy,repd})
  local f = nn.Select(2,1)(focal)
  local repf = nn.Replicate(2,2)(f)
  local repf = nn.Replicate(numPt,3)(repf)
  local proj = nn.CMulTable()({proj,repf})
  local proj = nn.AddConstant(inputRes/2)(proj)

  -- Final model
  local model = nn.gModule({inp}, {repos, trans, focal, proj})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

return M