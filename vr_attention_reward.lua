local mynn = require('mynn')
local logger = require('logger')()
local VRAttentionReward, parent = torch.class('mynn.VRAttentionReward', 'nn.Criterion')

function VRAttentionReward:__init(model, scale, criterion)
    parent.__init(self)
    self.model = model -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function VRAttentionReward:updateOutput(input, target)
    -- Input is a table
    -- 1. Final output
    -- 2. Groundtruth item ID
    -- 3. Collection of item IDs
    -- 4. Baseline reward (attention)
    -- 5. Baseline reward (counting)
    local output = input[1]
    local GTItemId = input[2]
    local itemId = input[3]
    itemId = itemId:reshape(GTItemId:size(1), itemId:numel() / GTItemId:size(1))

    local decoder = self.model.moduleMap['decoder']
    local attentionIdx = {}
    for t, decoderUnit in ipairs(decoder.replicas) do
        local _, idx = decoderUnit.moduleMap['attention'].output:max(2)
        table.insert(attentionIdx, idx)
    end
    local mse = (output - target):cmul(output - target):sum(2):reshape(output:size(1))
    mse = (mse / #attentionIdx):reshape(mse:size(1), 1)
    self.reward = torch.Tensor(input[1]:size(1), #attentionIdx):zero()

    local verb = false
    for n = 1, input[1]:size(1) do
        local counted = {}
        for i = 1, itemId:size(2) do
            counted[i] = false
        end
        for t = 1, #attentionIdx do
            local idx = attentionIdx[t][n][1]
            local attItemId = itemId[n][idx]
            if verb then
                io.write(string.format(' att idx %d', idx))
                io.write(string.format(' gt item id %d', GTItemId[n]))
                io.write(string.format(' att item id %d', attItemId))
                io.write(string.format(' counted %s', counted[idx]))
                io.write(string.format(' bin %.3f', binaryOutput[t][n][1]))
            end

            -- Attention reward
            if attItemId == GTItemId[n] then
                self.reward[n][t] = self.reward[n][t] + 1
            else
                self.reward[n][t] = self.reward[n][t] - 1
            end
            if verb then io.write('\n') end
        end
    end

    if self.sizeAverage then
        self.output = - self.reward:mean()
    else
        self.output = - self.reward:sum()
    end

    return self.output
end

-------------------------------------------------------------------------------
function VRAttentionReward:updateGradInput(inputTable, target)
    -- attention rewards
    local baselineAtt = self:toBatch(inputTable[4], 1)
    self.vrReward = self.vrReward or self.reward.new()
    self.vrReward:resizeAs(self.reward):copy(self.reward)
    self.vrReward:add(-1, baselineAtt)
    if self.sizeAverage then
        self.vrReward:div(inputTable[1]:size(1))
    end

    -- broadcast reward to modules
    local decoder = self.model.moduleMap['decoder']
    for t, rep in ipairs(decoder.replicas) do
        decoder.replicas[t].moduleMap['attention']:reinforce(
            self.vrReward[{{}, t}]:clone())
    end
    
    -- zero gradInput (this criterion has no gradInput for class pred)
    self.gradInput[1] = torch.Tensor(inputTable[1]:size()):zero()
    self.gradInput[2] = torch.Tensor(inputTable[2]:size()):zero()
    self.gradInput[3] = torch.Tensor(inputTable[3]:size()):zero()

    -- learn the baseline reward
    self.gradInput[4] = self.criterion:backward(baselineAtt, self.reward)

    return self.gradInput
end

-------------------------------------------------------------------------------
function VRAttentionReward:type(type)
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
