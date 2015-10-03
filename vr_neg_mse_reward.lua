-------------------------------------------------------------------------------
--[[ VRNegMSEReward ]]--
-- Variance reduced negative mean-squared error reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is negative distance.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRNegMSEReward, nn.SelectTable(-1))
-------------------------------------------------------------------------------
local mynn = require('mynn')
local VRNegMSEReward, parent = torch.class('mynn.VRNegMSEReward', 'nn.Criterion')

function VRNegMSEReward:__init(reinforceUnits, scale, criterion)
   parent.__init(self)
   self.reinforceUnits = reinforceUnits
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function VRNegMSEReward:updateOutput(input, target)
   assert(torch.type(input) == 'table')
   local input = self:toBatch(input[1], 1)
   self.reward = -(input - target):cmul(input - target):sum(2):mul(self.scale):reshape(input:size(1))
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

-------------------------------------------------------------------------------
function VRNegMSEReward:updateGradInput(inputTable, target)
   local input = self:toBatch(inputTable[1], 1)
   local baseline = self:toBatch(inputTable[2], 1)
   
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- print(self.vrReward)
   -- broadcast reward to modules
   -- print(torch.cat(self.reward, baseline, 2))
   if type(self.reinforceUnits) == 'table' then
      for i, unit in ipairs(self.reinforceUnits) do
         unit:reinforce(self.vrReward)
      end
   else
      self.reinforceUnits:reinforce(self.vrReward)
   end

   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(input):zero()
   self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)
   
   -- learn the baseline reward
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
   return self.gradInput
end

-------------------------------------------------------------------------------
function VRNegMSEReward:type(type)
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
