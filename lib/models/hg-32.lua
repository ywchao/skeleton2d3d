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
  local hmap = nn.Identity()()
  local proj = nn.Identity()()

  -- Initial processing of the image
  local in1 = cudnn.SpatialConvolution(numPt,32,1,1,1,1,0,0)(hmap)
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

  -- Back project 2d points
  local repd = nn.Replicate(2,3)(depth)
  local repf = nn.Replicate(numPt,2)(focal)
  local repf = nn.Replicate(2,3)(repf)
  local pt2d = nn.AddConstant(-32)(proj) -- TODO: get -32 from opt
  local pt3d = nn.CMulTable()({pt2d,repd})
  local pt3d = nn.CDivTable()({pt3d,repf})
  local viewd = nn.View(-1,numPt,1)(depth)
  local pt3d = nn.JoinTable(3)({pt3d,viewd})

  -- Output limb Length
  local split = nn.SplitTable(2)(pt3d)
  local j1 = nn.SelectTable(1)(split)
  local j2 = nn.SelectTable(2)(split)
  local j3 = nn.SelectTable(3)(split)
  local j4 = nn.SelectTable(4)(split)
  local j5 = nn.SelectTable(5)(split)
  local j6 = nn.SelectTable(6)(split)
  local j7 = nn.SelectTable(7)(split)
  local j8 = nn.SelectTable(8)(split)
  local j9 = nn.SelectTable(9)(split)
  local j10 = nn.SelectTable(10)(split)
  local j11 = nn.SelectTable(11)(split)
  local j12 = nn.SelectTable(12)(split)
  local j13 = nn.SelectTable(13)(split)
  local j14 = nn.SelectTable(14)(split)
  local j15 = nn.SelectTable(15)(split)
  local j16 = nn.SelectTable(16)(split)
  local j17 = nn.SelectTable(17)(split)
  local l1 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j1,j2})))))
  local l2 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j2,j3})))))
  local l3 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j3,j4})))))
  local l4 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j1,j5})))))
  local l5 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j5,j6})))))
  local l6 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j6,j7})))))
  local l7 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j1,j8})))))
  local l8 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j8,j9})))))
  local l9 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j9,j10})))))
  local l10 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j10,j11})))))
  local l11 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j9,j12})))))
  local l12 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j12,j13})))))
  local l13 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j13,j14})))))
  local l14 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j9,j15})))))
  local l15 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j15,j16})))))
  local l16 = nn.View(-1,1)(nn.Sqrt()(nn.Sum(2)(nn.Power(2)(nn.CSubTable()({j16,j17})))))
  local len = nn.JoinTable(2)({l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16})

  -- Final model
  local model = nn.gModule({hmap, proj}, {depth, focal, len})

  -- Zero the gradients; not sure if this is necessary
  model:zeroGradParameters()

  return model
end

return M