local mynn = require('mynn')
local logger = require('logger')()
local DoubleCountingCriterion, parent = torch.class('mynn.DoubleCountingCriterion', 'nn.Criterion')

-------------------------------------------------------------------------------
function DoubleCountingCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion or nn.BCECriterion()
    self.sizeAverage = true
    self.gradInput = {}
end

-------------------------------------------------------------------------------
function DoubleCountingCriterion:updateOutput(input, target)
    local numItems
    self.inputReshape = input[1]:reshape(input[1]:size(1), input[1]:size(2))
    self.gtGrid = input[2]
    _, self.attentionIdx = torch.max(input[3], 3)
    numItems = torch.max(self.gtGrid)
    self.labels = torch.Tensor(self.inputReshape:size()):zero()
    for n = 1, self.inputReshape:size(1) do
        local seen = {}
        for i = 1, numItems do
            seen[i] = 0
        end
        for t = 1, self.inputReshape:size(2) do
            local idx = self.attentionIdx[n][t][1]
            local grid = self.gtGrid[n][idx]
            self.labels[n][t] = 1 - seen[grid]
            seen[grid] = 1
            if verb then io.write('\n') end
        end
    end
    -- print(self.inputReshape, self.gtGrid, self.labels)
    self.output = self.criterion:forward(self.inputReshape, self.labels)
    -- print(self.output)
    return self.output
end

-------------------------------------------------------------------------------
function DoubleCountingCriterion:updateGradInput(input, target)
    self.gradInput[1] = self.criterion:backward(self.inputReshape, self.labels)
    self.gradInput[2] = torch.Tensor(input[2]:size()):zero()
    self.gradInput[3] = torch.Tensor(input[3]:size()):zero()
    return self.gradInput
end

-------------------------------------------------------------------------------
function DoubleCountingCriterion:type(type)
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
