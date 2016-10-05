--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('skeleton2d3d.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      -- local dataset = Dataset(opt, split)
      -- loaders[split] = M.DataLoader(dataset, opt, split)
      -- loaders[split] = M.DataLoader(opt, split)
      local Dataset = require('lib/datasets/' .. opt.dataset)
      local dataset = Dataset(opt, split)
      loaders[split] = M.DataLoader(dataset, opt, split)
   end

   -- return table.unpack(loaders)
   return loaders
end

function DataLoader:__init(dataset, opt, split)
-- function DataLoader:__init(opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      -- require('lib/datasets/' .. opt.dataset)
      -- We should have initialize dataset in creat(). This is currently not
      -- possible since the used hdf5 library will throw errors if we do that.
      local Dataset = require('lib/datasets/' .. opt.dataset)
      -- dataset = Dataset(opt, split)
   end
   local function main(idx)
      -- This matters due to the thread-dependent randomness from data synthesis
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      return dataset:size()
   end

   local threads, outputs = Threads(opt.nThreads, init, main)
   self.threads = threads
   self.__size = outputs[1][1]
   self.batchSize = opt.batchSize
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:sizeDataset()
   return self.__size
end

function DataLoader:run(kwargs)
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm
   assert(kwargs ~= nil and kwargs.train ~= nil)
   if kwargs.train then
      perm = torch.randperm(size)
   else
      batchSize = 1
      perm = torch.range(1, size)
   end

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices)
               local sz = indices:size(1)
               local input, imageSize
               local depth, depthSize, focal
               local pose, poseSize
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx, kwargs.train)
                  if not input then
                     imageSize = sample.input:size():totable()
                     input = torch.FloatTensor(sz, unpack(imageSize))
                  end
                  if not depth then
                     depthSize = sample.depth:size():totable()
                     depth = torch.FloatTensor(sz, unpack(depthSize))
                     focal = torch.FloatTensor(sz, 1)
                     poseSize = sample.pose:size():totable()
                     pose = torch.FloatTensor(sz, unpack(poseSize))
                  end
                  input[i] = sample.input
                  depth[i] = sample.depth
                  focal[i] = sample.focal
                  pose[i] = sample.pose
               end
               collectgarbage()
               return {
                  index = indices:int(),
                  input = input,
                  depth = depth,
                  focal = focal,
                  pose = pose,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader