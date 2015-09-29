local mynn = require('mynn')
local logger = require('logger')()
local AttentionCriterion, parent = torch.class('mynn.AttentionCriterion', 'nn.Criterion')

function AttentionCriterion:__init(decoder, criterion)
    parent.__init(self)
    self.decoder = decoder
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function AttentionCriterion:updateOutput(input, target)
    -- Input is a table
    -- 1. Attention
    -- 2. Groundtruth item ID
    -- 3. Collection of item IDs
    self.inputReshape = input[1]
    local GTItemId = input[2]
    local itemId = input[3]
    itemId = itemId:reshape(GTItemId:size(1), itemId:numel() / GTItemId:size(1))

    local attentionIdx = {}
    -- print(self.inputReshape:size())
    -- print(GTItemId:size())
    -- print(itemId:size())
    for t = 1, self.inputReshape:size(2) do
        -- print(t)
        local _, idx = self.inputReshape[{{}, t, {}}]:max(2)
        table.insert(attentionIdx, idx)
    end

    self.labels = torch.Tensor(
        self.inputReshape:size(1), #attentionIdx, itemId:size(2)):zero()
    for n = 1, self.inputReshape:size(1) do
        -- local count = {}
        -- for i = 1, 4 do
        --     count[i] = 0
        -- end
        -- for i = 1, itemId:size(2) do
        --     count[itemId[n][i]] = count[itemId[n][i]] + 1
        -- end
        -- if count[GTItemId[n]] > 0 then
        --     for i = 1, itemId:size(2) do
        --         if itemId[n][i] == GTItemId[n] then
        --             self.labels[{n, {}, i}] = torch.Tensor(
        --                 #attentionIdx):fill(1 / count[GTItemId[n]])
        --         end
        --     end
        -- else
        --     self.labels[{n, {}, {}}] = torch.Tensor(
        --         #attentionIdx, itemId:size(2)):fill(1 / itemId:size(2))
        -- end
        for i = 1, itemId:size(2) do
            if itemId[n][i] == GTItemId[n] then
                self.labels[{n, {}, i}] = torch.Tensor(
                    #attentionIdx):fill(1.0)
            else
                self.labels[{n, {}, i}] = torch.Tensor(
                    #attentionIdx):fill(0.01)
            end
        end
    end
    self.output = self.criterion:forward(self.inputReshape, self.labels)

    return self.output
end

-------------------------------------------------------------------------------
function AttentionCriterion:updateGradInput(inputTable, target)
    local gradInput = self.criterion:backward(self.inputReshape, self.labels)

    self.gradInput[1] = gradInput
    self.gradInput[2] = torch.Tensor(inputTable[2]:size()):zero()
    self.gradInput[3] = torch.Tensor(inputTable[3]:size()):zero()

    -- broadcast reward to modules
    local decoder = self.decoder
    for t, rep in ipairs(decoder.replicas) do
        decoder.replicas[t].moduleMap['attention']:reinforce(torch.Tensor(self.inputReshape:size(1)):zero())
    end
    return self.gradInput
end

-------------------------------------------------------------------------------
function AttentionCriterion:type(type)
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
