local mynn = require('mynn')
local logger = require('logger')()
local CountingCriterion, parent = torch.class('mynn.CountingCriterion', 'nn.Criterion')

function CountingCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion or nn.MSECriterion()
    self.sizeAverage = true
    self.gradInput = {}
end

-------------------------------------------------------------------------------
function CountingCriterion:updateOutput(input, target)
    -- print(input:size())
    -- print(self.recallerOutput.output:size())
    self.inputReshape = input[1]:reshape(input[1]:size(1), input[1]:size(2))
    self.recallerOutput = input[2]:reshape(input[1]:size(1), input[1]:size(2))
    self.labels = torch.Tensor(self.inputReshape:size()):zero()
    local inputHard = (self.recallerOutput:gt(
        torch.Tensor(self.recallerOutput:size()):fill(0.5))):float()
    for t = 1, self.recallerOutput:size(2) do
        if t == 1 then
            self.labels[{{}, t}] = inputHard[{{}, t}]
        else
            self.labels[{{}, t}] = self.labels[{{}, t - 1}] + inputHard[{{}, t}]
        end
    end
    -- print(self.recallerOutput, inputHard, self.inputReshape, self.labels)
    self.output = self.criterion:forward(self.inputReshape, self.labels)
    -- print(self.output)
    return self.output
end

-------------------------------------------------------------------------------
function CountingCriterion:updateGradInput(input, target)
    self.gradInput[1] = self.criterion:backward(self.inputReshape, self.labels)
    self.gradInput[2] = torch.Tensor(self.recallerOutput:size()):zero()
    return self.gradInput
end

-------------------------------------------------------------------------------
function CountingCriterion:type(type)
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
