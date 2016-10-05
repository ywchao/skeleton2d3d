local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 skeleton2d3d training script')
  cmd:text()
  cmd:text('Options:')
  cmd:text(' ------------ General options --------------------')
  cmd:option('-dataset',           'h36m', 'Options: h36m')
  cmd:option('-data',       './data/h36m', 'Path to dataset')
  cmd:option('-manualSeed',             3, 'Manually set RNG seed')
  cmd:option('-GPU',                    1, 'Default preferred GPU')
  cmd:option('-expDir',           './exp', 'Directory in which to save/log experiments')
  cmd:option('-expID',          'default', 'Experiment ID')
  cmd:text(' ------------ Data options -----------------------')
  cmd:option('-nThreads',               4, 'Number of data loading threads')
  cmd:text(' ------------ Training options -------------------')
  cmd:option('-nEpochs',               50, 'Number of total epochs to run')
  cmd:option('-batchSize',             64, 'Training mini-batch size (1 = pure stochastic)')
  cmd:option('-weightFocal',            1, 'Loss weight for focal length')
  cmd:option('-weightLenPrior',         1, 'Loss weight for limb length prior')
  cmd:text(' ------------ Checkpointing options --------------')
  cmd:option('-resume',             false, 'Resume from the latest checkpoint in this directory')
  cmd:text(' ------------ Optimization  options --------------')
  cmd:option('-LR',                 0.001, 'Initial learning rate')
  cmd:option('-weightDecay',            0, 'Weight decay')
  cmd:text(' ------------ Model options ----------------------')
  cmd:option('-netType',          'hg-32', 'Options: ')
  cmd:option('-inputRes',              64, 'Input image resolution')
  cmd:text()

  local opt = cmd:parse(arg or {})

  opt.expDir = paths.concat(opt.expDir, opt.dataset)
  opt.save = paths.concat(opt.expDir, opt.expID)

  if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
    cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
  end

  return opt
end

return M
