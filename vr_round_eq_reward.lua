-------------------------------------------------------------------------------
--[[ VRRoundEqReward ]]--
-- Variance reduced negative mean-squared error reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is negative distance.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VREqReward, nn.SelectTable(-1))
-------------------------------------------------------------------------------
local mynn = require('mynn')
local VRRoundEqReward, parent = torch.class('mynn.VRRoundEqReward', 'nn.Criterion')

function VRRoundEqReward:__init(module, scale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function VRRoundEqReward:updateOutput(input, target)
   assert(torch.type(input) == 'table')
   local input = self:toBatch(input[1], 1)
   self.reward =  torch.round(input):eq(target):float()
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output / input:size(1)
   end
   return self.output
end

-------------------------------------------------------------------------------
function VRRoundEqReward:updateGradInput(inputTable, target)
   local input = self:toBatch(inputTable[1], 1)
   local baseline = self:toBatch(inputTable[2], 1)
   
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- broadcast reward to modules
   self.module:reinforce(self.vrReward)
   
   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(input):zero()
   self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)
   
   -- learn the baseline reward
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
   return self.gradInput
end

-------------------------------------------------------------------------------
function VRRoundEqReward:type(type)
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
