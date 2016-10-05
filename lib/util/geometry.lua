
local eval = require 'lib/util/eval'

local M = {}

function M.angle2dcm(r1, r2, r3, S)
  S = S or 'zyx'

  local dcm = torch.zeros(3,3)
  local cang = torch.cos(torch.Tensor{r1,r2,r3})
  local sang = torch.sin(torch.Tensor{r1,r2,r3})

  -- now only supports 'zyx'
  assert(S:lower() == 'zyx')

  if S:lower() == 'zyx' then
    dcm[1][1] = cang[2] * cang[1]
    dcm[1][2] = cang[2] * sang[1]
    dcm[1][3] = -sang[2]
    dcm[2][1] = sang[3] * sang[2] * cang[1] - cang[3] * sang[1]
    dcm[2][2] = sang[3] * sang[2] * sang[1] + cang[3] * cang[1]
    dcm[2][3] = sang[3] * cang[2]
    dcm[3][1] = cang[3] * sang[2] * cang[1] + sang[3] * sang[1]
    dcm[3][2] = cang[3] * sang[2] * sang[1] - sang[3] * cang[1]
    dcm[3][3] = cang[3] * cang[2]
  end

  return dcm
end

function M.camProject(P, R, T, f, c)
-- input
--	P:	N x 3
--	R:	3 x 3
--	T:	1 x 3
--	f:	1 x 2
--	c:	1 x 2
-- output
--	p:	N x 2
--	D:	N x 1
  local N = P:size(1)
  local X = R * torch.csub(P:t(),T:expand(N,3):t())
  local p = torch.cdiv(X[{{1,2},{}}], X[{{3},{}}]:expand(2,N))
  local p = torch.cmul(f:expand(N,2), p:t()) + c:expand(N,2)
  local D = X[{{3}}]:t()
  return p, D, X
end

function M.backProjectHMDM(hm, dm, f)
-- input
--	hm:	3 x h x w
--	dm: N x h x w
  local pt = eval.getPreds(hm:view(1,unpack(hm:size():totable())))[1]
  local c = torch.Tensor{hm:size(3),hm:size(2)}:div(2)
  local P = torch.zeros(3,pt:size(1))
  for j = 1, pt:size(1) do
    local x = pt[j][1]
    local y = pt[j][2]
    local d = dm[j][y][x]
    P[1][j] = (x - c[1]) * d / f
    P[2][j] = (y - c[2]) * d / f
    P[3][j] = d
  end
  return P
end

return M