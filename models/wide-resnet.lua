--  Wide Residual Network
--  This is an implementation of the wide residual networks described in:
--  "Wide Residual Networks", http://arxiv.org/abs/1605.07146
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
local utils = paths.dofile'utils.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.num_classes)
   assert(opt and opt.widen_factor)

   local function Dropout()
      return nn.Dropout(opt and opt.dropout or 0,nil,true)
   end

   local depth = opt.depth

   local blocks = {}
   
   local function wide_basic(nInputPlane, nOutputPlane, stride, first)
      local card = 1

      if not first then
          nInputPlane = 4*nInputPlane
      end

      print('---')
      print(nInputPlane)
      print(4*nOutputPlane)

      local nBottleneckPlane = nOutputPlane
      local conv_params = {
         {nInputPlane,card*nBottleneckPlane,1,1,1,1,0,0},
         {card*nBottleneckPlane,card*nBottleneckPlane,3,3,stride,stride,1,1},
         {card*nBottleneckPlane,4*nBottleneckPlane,1,1,1,1,0,0},
      }

      local block = nn.Sequential()
      local convs = nn.Sequential()     

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = (nInputPlane == nOutputPlane and not first) and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            convs:add(Convolution(table.unpack(v)))
         else
            convs:add(SBatchNorm(card*nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(table.unpack(v)))
         end
      end
     
      --convs:add(nn.Mul())
 
      local shortcut = (nInputPlane == 4*nOutputPlane and not first) and
         nn.Identity() or
         Convolution(nInputPlane,4*nOutputPlane,1,1,stride,stride,0,0)
     
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride, first)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride, first))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1, false))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2')
      local n = (depth - 2) / 9

      local k = opt.widen_factor
      local nStages = torch.Tensor{64, 64, 128, 256}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1, true)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2, false)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2, false)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(4*nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(4*nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(4*nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)
   
   -- model:get(1).gradInput = nil

   return model
end

return createModel
