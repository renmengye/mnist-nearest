local mynn = require('mynn')
local logger = require('logger')()
local DoubleCountingCriterion, parent = torch.class('mynn.DoubleCountingCriterion', 'nn.Criterion')

function DoubleCountingCriterion:__init(decoder, criterion)
    parent.__init(self)
    self.decoder = decoder
    self.criterion = criterion or nn.BCECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function DoubleCountingCriterion:updateOutput(input, target)
    local attentionIdx = {}
    for t, decoderUnit in ipairs(self.decoder.replicas) do
        local _, idx = decoderUnit.moduleMap['attention'].output:max(2)
        table.insert(attentionIdx, idx)
    end
    self.inputReshape = input:reshape(input:size(1), input:size(2))
    self.labels = torch.Tensor(self.inputReshape:size()):zero()
    for n = 1, input:size(1) do
        local seen = {}
        for i = 1, 9 do
            seen[i] = 0
        end
        for t = 1, #attentionIdx do
            local idx = attentionIdx[t][n][1]
            self.labels[n][t] = 1 - seen[idx]
            seen[idx] = 1
            if verb then io.write('\n') end
        end
    end
    -- print(self.inputReshape, self.labels)
    self.output = self.criterion:forward(self.inputReshape, self.labels)
    -- print(self.output)
    return self.output
end

-------------------------------------------------------------------------------
function DoubleCountingCriterion:updateGradInput(input, target)
    self.gradInput = self.criterion:backward(self.inputReshape, self.labels)
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
