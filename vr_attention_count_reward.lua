local mynn = require('mynn')
local logger = require('logger')()
local VRAttentionCountReward, parent = torch.class('mynn.VRAttentionCountReward', 'nn.Criterion')

function VRAttentionCountReward:__init(model, scale, criterion)
    parent.__init(self)
    self.model = model -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = {torch.Tensor()}
end

-------------------------------------------------------------------------------
function VRAttentionCountReward:updateOutput(input, target)
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
    local binaryOutput = {}
    for t, decoderUnit in ipairs(decoder.replicas) do
        local _, idx = decoderUnit.moduleMap['attention'].output:max(2)
        table.insert(attentionIdx, idx)
        table.insert(binaryOutput, decoderUnit.moduleMap['binaryOutput'].output)
    end
    local mse = (output - target):cmul(output - target):sum(2):reshape(output:size(1))
    mse = (mse / #attentionIdx):reshape(mse:size(1), 1)
    self.attentionReward = torch.Tensor(input[1]:size(1), #attentionIdx):zero()
    self.countingReward = torch.Tensor(input[1]:size(1), #attentionIdx):zero()

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
                self.attentionReward[n][t] = self.attentionReward[n][t] + 1
            else
                self.attentionReward[n][t] = self.attentionReward[n][t] - 1
            end

            -- if not counted[idx] and 
            --     attItemId == GTItemId[n] and 
            --     binaryOutput[t][n][1] == 1 then
            --     self.countingReward[n][t] = self.countingReward[n][t] + 1
            --     counted[idx] = true
            --     if verb then io.write(' + 2') end
            -- else
            --     -- Penalty for counting wrong
            --     if attItemId ~= GTItemId[n] and 
            --         binaryOutput[t][n][1] == 1 then
            --         self.countingReward[n][t] = self.countingReward[n][t] - 0.5
            --         if verb then io.write(' - 0.5') end
            --     end

            --     -- Penalty for double counting
            --     if counted[idx] and 
            --         attItemId == GTItemId[n] and 
            --         binaryOutput[t][n][1] == 1 then
            --         self.countingReward[n][t] = self.countingReward[n][t] - 0.5
            --         if verb then io.write(' - 0.5') end
            --     end
            -- end

            -- Counting reward
            if not counted[idx] and binaryOutput[t][n][1] == 1 and attItemId == GTItemId[n] then
                self.countingReward[n][t] = self.countingReward[n][t] + 0.5
                counted[idx] = true
            elseif counted[idx] and binaryOutput[t][n][1] == 1 and attItemId == GTItemId[n] then
                self.countingReward[n][t] = self.countingReward[n][t] - 0.5
            end
            if verb then io.write('\n') end
        end
    end

    if self.sizeAverage then
        self.output = - self.attentionReward:mean() - self.countingReward:mean()
    else
        self.output = - self.attentionReward:sum() - self.countingReward:sum()
    end

    return self.output
end

-------------------------------------------------------------------------------
function VRAttentionCountReward:updateGradInput(inputTable, target)
    -- attention rewards
    local baselineAtt = self:toBatch(inputTable[4], 1)
    self.vrAttReward = self.vrAttReward or self.attentionReward.new()
    self.vrAttReward:resizeAs(self.attentionReward):copy(self.attentionReward)
    self.vrAttReward:add(-1, baselineAtt)
    if self.sizeAverage then
        self.vrAttReward:div(inputTable[1]:size(1))
    end

    -- counting rewards
    local baselineCnt = self:toBatch(inputTable[5], 1)
    self.vrCntReward = self.vrCntReward or self.countingReward.new()
    self.vrCntReward:resizeAs(self.countingReward):copy(self.countingReward)
    self.vrCntReward:add(-1, baselineCnt)
    if self.sizeAverage then
        self.vrCntReward:div(inputTable[1]:size(1))
    end

    -- print(baselineAtt:mean(), self.attentionReward:mean())
    -- print(baselineCnt:mean(), self.countingReward:mean())

    -- broadcast reward to modules
    local decoder = self.model.moduleMap['decoder']
    for t, rep in ipairs(decoder.replicas) do
        decoder.replicas[t].moduleMap['binaryOutput']:reinforce(
            self.vrCntReward[{{}, t}]:clone())
        decoder.replicas[t].moduleMap['attention']:reinforce(
            self.vrAttReward[{{}, t}]:clone())
    end
    
    -- zero gradInput (this criterion has no gradInput for class pred)
    self.gradInput[1] = torch.Tensor(inputTable[1]:size()):zero()
    self.gradInput[2] = torch.Tensor(inputTable[2]:size()):zero()
    self.gradInput[3] = torch.Tensor(inputTable[3]:size()):zero()

    -- learn the baseline reward
    self.gradInput[4] = self.criterion:backward(baselineAtt, self.attentionReward)
    self.gradInput[5] = self.criterion:backward(baselineCnt, self.countingReward)

    return self.gradInput
end

-------------------------------------------------------------------------------
function VRAttentionCountReward:type(type)
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
