local mynn = require('mynn')
local logger = require('logger')()
local AttentionCriterion, parent = torch.class('mynn.AttentionCriterion', 'nn.Criterion')

function AttentionCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion or nn.MSECriterion()
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function AttentionCriterion:updateOutput(input, target)
    -- Input is a table
    -- 1. Soft attention
    -- 2. Hard attention
    -- 3. Groundtruth item ID
    -- 4. Collection of item IDs
    self.inputReshape = input[1]
    local GTItemId = input[3]
    local itemId = input[4]:reshape(
        GTItemId:size(1), input[4]:numel() / GTItemId:size(1))
    local attentionIdx = {}
    for t = 1, self.inputReshape:size(2) do
        local _, idx = self.inputReshape[{{}, t, {}}]:max(2)
        table.insert(attentionIdx, idx)
    end
    self.labels = torch.Tensor(
        self.inputReshape:size(1), #attentionIdx, itemId:size(2)):zero()
    for n = 1, self.inputReshape:size(1) do
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
    -- Pretend that we only know 10% of the groundtruth object of interst.
    -- local N
    -- if self.labels:size(1) > 10 then
    --     N = torch.floor(self.labels:size(1) / 10)
    -- else
    --     N = 1
    -- end    
    -- local gradInput = self.criterion:backward(self.inputReshape[{{1, N}}], 
    --     self.labels[{{1, N}}])
    -- self.gradInput[1] = torch.Tensor(inputTable[1]:size()):zero()
    -- self.gradInput[1][{{1, N}}]:copy(gradInput)
    local gradInput = self.criterion:backward(self.inputReshape, self.labels)
    self.gradInput[1] = gradInput
    self.gradInput[2] = torch.Tensor(inputTable[2]:size()):zero()
    self.gradInput[3] = torch.Tensor(inputTable[3]:size()):zero()
    self.gradInput[4] = torch.Tensor(inputTable[4]:size()):zero()

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
