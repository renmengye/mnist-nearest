local mynn = require('mynn')
local logger = require('logger')()
local RelatednessCriterion, parent = torch.class('mynn.RelatednessCriterion', 'nn.Criterion')

function RelatednessCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion or nn.BCECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function RelatednessCriterion:updateOutput(input, target)
    -- Input is a table
    -- 1. Relatedness  N x T x 1
    -- 2. Attention    N x T x M
    -- 3. Groundtruth item ID  N
    -- 4. Items N x M
    self.relatedness = input[1]
    self.attention = input[2]
    local GTItemId = input[3]
    local itemId = input[4]
    itemId = itemId:reshape(GTItemId:size(1), itemId:numel() / GTItemId:size(1))

    self.labels = torch.Tensor(self.relatedness:size()):zero()
    for t = 1, self.attention:size(2) do
        local _, idx = self.attention[{{}, t, {}}]:max(2)
        for n = 1, self.relatedness:size(1) do
            if itemId[n][idx[n][1]] == GTItemId[n] then
                self.labels[n][t][1] = 1.0
            end
        end
        -- self.labels[{{}, t, 1}] = itemId[idx]:eq(GTItemId):float()
        -- print(torch.cat(torch.cat(GTItemId, idx:float(), 2), self.labels[{{}, t, 1}], 2))
        -- print(self.labels)
    end
    self.output = self.criterion:forward(self.relatedness, self.labels)

    return self.output
end

-------------------------------------------------------------------------------
function RelatednessCriterion:updateGradInput(inputTable, target)
    local gradInput = self.criterion:backward(self.relatedness, self.labels)
    self.gradInput[1] = gradInput
    self.gradInput[2] = torch.Tensor(inputTable[2]:size()):zero()
    self.gradInput[3] = torch.Tensor(inputTable[3]:size()):zero()
    self.gradInput[4] = torch.Tensor(inputTable[4]:size()):zero()
    return self.gradInput
end

-------------------------------------------------------------------------------
function RelatednessCriterion:type(type)
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
