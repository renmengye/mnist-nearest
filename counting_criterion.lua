local mynn = require('mynn')
local logger = require('logger')()
local CountingCriterion, parent = torch.class('mynn.CountingCriterion', 'nn.Criterion')

function CountingCriterion:__init(recallerOutput, criterion)
    parent.__init(self)
    self.recallerOutput = recallerOutput
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function CountingCriterion:updateOutput(input, target)
    -- print(input:size())
    -- print(self.recallerOutput.output:size())
    local recallerOutput = self.recallerOutput.output:reshape(
        input:size(1), input:size(2))
    self.inputReshape = input:reshape(input:size(1), input:size(2))
    self.labels = torch.Tensor(self.inputReshape:size()):zero()
    local inputHard = (recallerOutput:gt(torch.Tensor(recallerOutput:size()):fill(0.5))):float()
    for t = 1, recallerOutput:size(2) do
        if t == 1 then
            self.labels[{{}, t}] = inputHard[{{}, t}]
        else
            self.labels[{{}, t}] = self.labels[{{}, t - 1}] + inputHard[{{}, t}]
        end
    end
    -- print(recallerOutput, inputHard, self.inputReshape, self.labels)
    self.output = self.criterion:forward(self.inputReshape, self.labels)
    -- print(self.output)
    return self.output
end

-------------------------------------------------------------------------------
function CountingCriterion:updateGradInput(input, target)
    self.gradInput = self.criterion:backward(self.inputReshape, self.labels)
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
