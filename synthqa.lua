local torch = require('torch')
local logger = require('logger')()
local table_utils = require('table_utils')
local imageqa = require('imageqa')
local lstm = require('lstm')
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local constant = require('constant')
local gradient_stopper = require('gradient_stopper')
local lazy_sequential = require('lazy_sequential')
local lazy_gmodule = require('lazy_gmodule')
local batch_reshape = require('batch_reshape')
local vr_neg_mse_reward = require('vr_neg_mse_reward')
local vr_round_eq_reward = require('vr_round_eq_reward')
local utils = require('utils')
local nntrainer = require('nntrainer')
local nnevaluator = require('nnevaluator')
local nnserializer = require('nnserializer')
local reinforce_container = require('reinforce_container')
local optim_pkg = require('optim')
local synthqa = {}

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

-------------------------------------------------------------------------------
synthqa.OBJECT = {
    'circle',
    'square',
    'triangle'
}

synthqa.X = {
    'top', 
    'middle', 
    'bottom'
}

synthqa.Y = {
    'left', 
    'center', 
    'right'
}

synthqa.COLOR = {
    'red', 
    'orange', 
    'yellow', 
    'green', 
    'blue', 
    'purple', 
    'black'
}

synthqa.RELATION = {
    'above', 
    'below', 
    'left_of', 
    'right_of'
}

synthqa.WORDS = {
    'what',
    'how',
    'many',
    'color',
    'is'
}

synthqa.NUMBER = {
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
}

synthqa.NUM_GRID = #synthqa.X * #synthqa.Y
synthqa.OBJECT_SIZE_AVG = 0.1
synthqa.OBJECT_SIZE_STD = 0.03
synthqa.X_STD = 0.05
synthqa.Y_STD = 0.05

-------------------------------------------------------------------------------
function combine(tables)
    local combined = {}
    local count = 1
    for i, t in ipairs(tables) do
        for j, v in ipairs(t) do
            combined[v] = count
            count = count + 1
        end
    end
    return combined
end

local imageSize = {300, 300}

-------------------------------------------------------------------------------
function synthqa.genHowManyObject(N)
    local dataset = {}
    for i = 1, N do
        local objectOfInterest = i % #synthqa.OBJECT + 1
        local question = string.format(
            'how many %s', synthqa.OBJECT[objectOfInterest])
        local items = {}
        local numObj = {}

        -- Empty object is encoded as the largest object ID.
        for j = 1, #synthqa.OBJECT + 1 do
            numObj[j] = 0
        end
        for j = 1, synthqa.NUM_GRID do
            local objCat = torch.ceil(torch.uniform() * (#synthqa.OBJECT + 1))
            numObj[objCat] = numObj[objCat] + 1
            table.insert(items, {category = objCat, color = 1, grid = j})
        end
        local answer = synthqa.NUMBER[numObj[objectOfInterest] + 1]
        logger:logInfo(
            string.format(
                'N1: %d, N2: %d, N3: %d, N: %d', 
                numObj[1], numObj[2], numObj[3], 
                numObj[1] + numObj[2] + numObj[3]), 2)
        logger:logInfo(
            string.format(
                'Q: %s, A: %s', 
                question, answer), 2)
        logger:logInfo(table.tostring(items), 2)
        table.insert(dataset, {
            question = question,
            answer = answer,
            items = items
        })
    end
    return dataset
end

-------------------------------------------------------------------------------
function synthqa.encodeItems(allItems)
    -- Category ID (1)
    -- Color ID (1)
    -- X, Y coordinates (2)
    -- local colorIdict = imageqa.invertDict(COLOR)
    -- local objTypeIdict = imageqa.invertDict(OBJECT)
    local function getCoord(grid, noise)
        if noise == nil then
            noise = true
        end
        -- Randomly sample with gaussian noise
        local yCenter = torch.floor((grid - 1) / 3)  / 3 + (1 / 6)
        local xCenter = (grid - 1) % 3 / 3 + (1 / 6)
        if noise then
            logger:logInfo(
                string.format(
                    'Grid %d, X center %f Y center %f', 
                    grid, xCenter, yCenter), 2)
            return torch.Tensor({torch.normal(xCenter, synthqa.X_STD),
                    torch.normal(yCenter, synthqa.Y_STD),
                    torch.normal(
                        synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD),
                    torch.normal(
                        synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD)})
        else
            return torch.Tensor(
                {xCenter, yCenter, 
                synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_AVG})
        end
    end
    local numDim = 1 + 1 + 4
    local result = torch.Tensor(#allItems, synthqa.NUM_GRID * numDim):zero()
    for i, items in ipairs(allItems) do
        if #items > 0 then
            local itemShuffle = torch.randperm(#items)
            -- Shuffle items
            for j = 1, #items do
                local item = items[itemShuffle[j]]
                local itemCount = 0
                for key, value in pairs(item) do
                    if key == 'category' then
                        result[{i, (j - 1) * numDim + 1}] = value
                    elseif key == 'color' then
                        result[{i, (j - 1) * numDim + 2}] = value
                    elseif key == 'grid' then
                        result[
                        {i, {(j - 1) * numDim + 3, (j - 1) * numDim + 6}}] = 
                        getCoord(value)
                    end
                end
            end
        end
    end
    return result
end

-------------------------------------------------------------------------------
synthqa.dict = combine({
            synthqa.OBJECT, 
            synthqa.X, 
            synthqa.Y, 
            synthqa.COLOR, 
            synthqa.RELATION, 
            synthqa.WORDS,
            synthqa.NUMBER
        })
synthqa.idict = imageqa.invertDict(synthqa.dict)

-------------------------------------------------------------------------------
function synthqa.prep(rawData, objective)
    if objective == nil then
        objective = 'classification'
    end
    logger:logInfo(table.tostring(synthqa.dict), 2)
    logger:logInfo(table.tostring(synthqa.idict), 2)
    local questions = {}
    local answers = {}
    local allItems = {}
    for i, entry in ipairs(rawData) do
        table.insert(questions, entry.question)
        table.insert(answers, entry.answer)
        table.insert(allItems, entry.items)
    end

    -- N x Q
    local questionIds = imageqa.encodeSentences(questions, synthqa.dict, true)
    -- N
    local answerIds = 
        imageqa.encodeSentences(answers, synthqa.dict, true):reshape(#answers)
    -- N x 54
    local itemIds = synthqa.encodeItems(allItems)
    logger:logInfo(questionIds, 2)
    logger:logInfo(answerIds, 2)
    logger:logInfo(itemIds:reshape(#answers, synthqa.NUM_GRID, 6), 2)

    -- N x (54 + Q)
    local data = torch.cat(itemIds, questionIds, 2)
    local labels = answerIds:long()
    if objective == 'regression' then
        labels = labels - synthqa.dict['0']
        labels = labels:float()
    end
    return data, labels
end

-------------------------------------------------------------------------------
function synthqa.checkExists(dataset, example)
    for i, example2 in ipairs(dataset) do
        if example.items.question == example2.items.question and 
            example.items.answer == example2.items.answer then
            local allSame = true
            for j, item2 in ipairs(example2.items) do
                local item = example.items[j]
                if item2.category ~= item.category or
                    item2.color ~= item.color or
                    item2.grid ~= item.grid then
                    allSame = false
                    break
                end
            end
            if allSame then
                return true
            end
        end
    end
    return false
end

-------------------------------------------------------------------------------
function synthqa.checkOverlap(dataset1, dataset2)
    local overlap = 0
    for i, example in ipairs(dataset2) do
        if synthqa.checkExists(dataset1, example) then
            overlap = overlap + 1
        end
    end
    logger:logInfo(string.format(
        '%.3f of dataset2 overlap with dataset1', overlap / #dataset2))
end

-------------------------------------------------------------------------------
function synthqa.createModel(params, training)
    -- params.objectEmbedDim
    -- params.colorEmbedDim
    -- params.questionLength
    -- params.wordEmbedDim
    -- params.lstmDim
    -- params.itemDim
    -- params.decoderSteps
    -- params.vocabSize
    -- params.numObject
    -- params.numColor
    -- params.attentionMechanism: 'soft' or 'hard'
    -- params.objective
    -- params.aggregatorWeights
    -- params.outputMapWeights
    
    if training == nil then
        training = false
    end

    -- Input
    local input = nn.Identity()()

    -- Items to attend
    local itemRawDim = params.objectEmbedDim + params.colorEmbedDim + 4
    local items = nn.Narrow(2, 1, 54)(input)
    local itemsReshape = nn.Reshape(9, 6)(items)
    local catId = nn.Select(3, 1)(itemsReshape)
    local catIdReshape = mynn.BatchReshape()(catId)
    catIdReshape.data.module.name = 'catIdReshape'
    local colorId = nn.Select(3, 2)(itemsReshape)
    local colorIdReshape = mynn.BatchReshape()(colorId)
    colorIdReshape.data.module.name = 'colorIdReshape'
    local coord = nn.Narrow(3, 3, 4)(itemsReshape)
    local coordReshape = mynn.BatchReshape(4)(coord)
    coordReshape.data.module.name = 'coordReshape'
    local catEmbed = nn.LookupTable(
        params.numObject, params.objectEmbedDim)(
        nn.GradientStopper()(catIdReshape))
    local colorEmbed = nn.LookupTable(
        params.numColor, params.colorEmbedDim)(
        nn.GradientStopper()(colorIdReshape))
    local itemsJoined = nn.JoinTable(2, 2)(
        {catEmbed, colorEmbed, coordReshape})
    local itemsJoinedReshape = mynn.BatchReshape(9, itemRawDim)(itemsJoined)
    itemsJoinedReshape.data.module.name = 'itemsJoinedReshape'

    -- Word Embeddings
    local wordIds = nn.Narrow(2, 55, params.questionLength)(input)
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(
        nn.GradientStopper()(wordIds))
    local wordEmbedSeq = nn.SplitTable(2)(wordEmbed)
    local constantState = mynn.Constant(params.lstmDim * 2, 0)(input)

    -- Encoder LSTM
    local encoderCore = lstm.createUnit(
        params.wordEmbedDim, params.lstmDim)
    local encoder = nn.RNN(
        encoderCore, params.questionLength)({wordEmbedSeq, constantState})
    local encoderStateSel = nn.SelectTable(
        params.questionLength)(encoder)
    encoder.data.module.name = 'encoder'

    -- Decoder dummy input
    local decoderInputConst = mynn.Constant(
        {params.decoderSteps, 1}, 0)(input)
    local decoderInputSplit = nn.SplitTable(2)(decoderInputConst)

    -- Decoder LSTM (1st layer)
    local decoderCore = lstm.createAttentionUnit(
        1, params.lstmDim, 9, itemRawDim, 0.1, 
        params.attentionMechanism, training)
    local decoder = nn.RNN(
        decoderCore, params.decoderSteps)(
        {decoderInputSplit, constantState, itemsJoinedReshape})
    decoder.data.module.name = 'decoder'

    local createBinaryInput = function()
        local input = nn.Identity()()
        local initRange = 0.1
        local inputHidden = nn.Narrow(2, params.lstmDim + 1, params.lstmDim)(input)
        local aggregate = nn.Linear(params.lstmDim, 1)(inputHidden)
        local sigmoid = nn.Sigmoid()(aggregate)
        local stochastic = nn.ReinforceBernoulli()(sigmoid)
        local unit = nn.gModule({input}, {stochastic})
        aggregate.data.module.weight:uniform(-initRange / 2, initRange / 2)
        aggregate.data.module.bias:uniform(-initRange / 2, initRange / 2)
        unit.moduleMap = {
            input = stochastic.data.module
        }
        unit.reinforceUnit = stochastic.data.module
        return unit
    end
    local binaryInput = nn.RNN(createBinaryInput(), params.decoderSteps, false)(decoder)
    binaryInput.data.module.name = 'binary'

    -- Decoder LSTM (2nd stochastic binary input layer)
    local aggregator = nn.RNN(
        lstm.createUnit(1, params.lstmDim), params.decoderSteps)(
        {binaryInput, constantState})
    if params.aggregatorWeights then
        local agg_w, agg_dl_dw = aggregator.data.module.core:getParameters()
        agg_w:copy(params.aggregatorWeights)
    end
    aggregator.data.module.name = 'aggregator'

    -- Classify answer
    local decoderOutputSel = nn.SelectTable(
        params.decoderSteps)(aggregator)
    local decoderOutputState = nn.Narrow(
        2, params.lstmDim + 1, params.lstmDim)(decoderOutputSel)

    local outputMap, final
    if params.objective == 'regression' then
        outputMap = nn.Linear(params.lstmDim, 1)(decoderOutputState)
        final = outputMap
    elseif params.objective == 'classification' then
        outputMap = nn.Linear(
            params.lstmDim, params.vocabSize)(decoderOutputState)
        local answerlog = nn.LogSoftMax()(outputMap)
        final = answerlog
    else
        logger:logFatal(string.format(
            'unknown training objective %s', params.objective))
    end

    if params.outputMapWeights then
        local ow, odldw = outputMap.data.module:getParameters()
        -- print(ow:size())
        -- print(params.outputMapWeights:size())
        ow:copy(params.outputMapWeights)
    end

    -- Build entire model
    -- Need MSECriterion for regression reward.
    local all, expectedReward
    if params.attentionMechanism == 'soft' then
        -- all = nn.LazyGModule({input}, {answer})
        all = nn.LazyGModule({input}, {final})
    elseif params.attentionMechanism == 'hard' then
        expectedReward = mynn.Weights(1)(input)
        local reinforceOut = nn.Identity()({final, expectedReward})
        all = nn.LazyGModule({input}, {final, reinforceOut})
    else
        logger:logFatal(string.format(
            'unknown attentionMechanism %s', params.attentionMechanism))
    end

    all:addModule('catEmbed', catEmbed)
    all:addModule('colorEmbed', colorEmbed)
    all:addModule('wordEmbed', wordEmbed)
    all:addModule('encoder', encoder)
    all:addModule('decoder', decoder)
    all:addModule('binaryInput', binaryInput)
    all:addModule('aggregator', aggregator)
    all:addModule('answer', outputMap)
    if params.attentionMechanism == 'hard' then
        all:addModule('expectedReward', expectedReward)
    end
    all:setup()

    -- Expand LSTMs
    encoder.data.module:expand()
    decoder.data.module:expand()
    binaryInput.data.module:expand()
    aggregator.data.module:expand()

    if params.attentionMechanism == 'soft' then
        if params.objective == 'regression' then
            all.criterion = nn.MSECriterion()
        elseif params.objective == 'classification' then
            all.criterion = nn.ClassNLLCriterion()
        else
            logger:logFatal(string.format(
                'unknown training objective %s', params.objective))
        end
        -- all.criterion = nn.CrossEntropyCriterion()
    elseif params.attentionMechanism == 'hard' then
        -- Setup criterions and rewards
        local reinforceUnits = {}
        for i = 1, params.decoderSteps do
            table.insert(
                reinforceUnits, decoder.data.module.replicas[i].reinforceUnit)
            table.insert(
                reinforceUnits, 
                binaryInput.data.module.replicas[i].reinforceUnit)
        end
        local rc = ReinforceContainer(reinforceUnits)

        if params.objective == 'regression' then
            all.criterion = nn.ParallelCriterion(true)
              :add(nn.ModuleCriterion(
                nn.MSECriterion(), nil, nn.Convert()))
              :add(nn.ModuleCriterion(
                mynn.VRNegMSEReward(rc), nil, nn.Convert()))
        elseif params.objective == 'classification' then
            all.criterion = nn.ParallelCriterion(true)
              :add(nn.ModuleCriterion(
                nn.ClassNLLCriterion(), nil, nn.Convert()))
              :add(nn.ModuleCriterion(
                nn.VRClassReward(rc), nil, nn.Convert()))
        else
            logger:logFatal(string.format(
                'unknown training objective %s', params.objective))
        end
    end

    if params.objective == 'regression' then
        all.decision = function(pred)
            local num = torch.round(pred)
            return num
        end
    elseif params.objective == 'classification' then
        all.decision = function(pred)
            local score, idx = pred:max(2)
            return idx
        end
    end

    return all
end

return synthqa
