local torch = require('torch')
local logger = require('logger')()
local tableUtils = require('table_utils')
local imageqa = require('imageqa')
local lstm = require('lstm')
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local constant = require('constant')
local weights = require('weights')
local lazy_sequential = require('lazy_sequential')
local batch_reshape = require('batch_reshape')
local utils = require('utils')
local synthqa = {}

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------

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

synthqa.numObj1 = {
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine'
}

synthqa.NUM_GRID = #synthqa.X * #synthqa.Y
synthqa.OBJECT_SIZE_AVG = 0.1
synthqa.OBJECT_SIZE_STD = 0.03
synthqa.X_STD = 0.05
synthqa.Y_STD = 0.05

----------------------------------------------------------------------

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

----------------------------------------------------------------------

function synthqa.genHowManyObject(N)
    -- Uniform distribution between 0 and 9
    local dataset = {}
    for i = 1, N do
        local numObj1 = torch.floor(torch.uniform() * #synthqa.numObj1)
        local objectOfInterest = i % #synthqa.OBJECT + 1
        local question = string.format(
            'how many %s', synthqa.OBJECT[objectOfInterest])
        local answer = synthqa.numObj1[numObj1 + 1]
        -- We shuffle a list of 1-9
        -- The first #numObj1 of grids are object of interest
        -- Then we have some numObj1 of other objects
        -- Then we have empty grids
        local sampleGrid = torch.randperm(synthqa.NUM_GRID)
        local items = {}
        if numObj1 > 0 then
            for j = 1, numObj1 do
                table.insert(items, {
                    category = objectOfInterest, 
                    grid = sampleGrid[j]
                })
            end
        end
        local numObj2 = torch.floor(
            torch.uniform() * (synthqa.NUM_GRID - numObj1))
        local nextObj = torch.ceil(torch.uniform() * 2)
        if numObj2 > 0 then
            for j = numObj1 + 1, numObj1 + numObj2 do
                table.insert(items, {
                    category = (objectOfInterest - 1 + nextObj) % 
                               #synthqa.OBJECT + 1,
                    grid = j 
                })
            end
        end
        local numObj3 = torch.floor(
            torch.uniform() * (synthqa.NUM_GRID - numObj1 - numObj2))
        if numObj3 > 0 then
            for j = numObj1 + numObj2 + 1, numObj1 + numObj2 + numObj3 do
                table.insert(items, {
                    category = (objectOfInterest - 1 + 2 * nextObj) % 
                               #synthqa.OBJECT + 1,
                    grid = j
                })
            end
        end
        logger:logInfo(
            string.format(
                'N1: %d, N2: %d, N3: %d, N: %d', 
                numObj1, numObj2, numObj3, numObj1 + numObj2 + numObj3), 2)
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

----------------------------------------------------------------------

function synthqa.encodeItems(items)
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
                    torch.normal(synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD),
                    torch.normal(
                        synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD)})
        else
            return torch.Tensor(
                {xCenter, yCenter, synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_AVG})
        end
    end
    local numDim = 1 + 1 + 4
    local result = torch.Tensor(#items, synthqa.NUM_GRID * numDim):zero()
    for i, example in ipairs(items) do
        local itemShuffle = torch.randperm(#example)
        for j = 1, #example do
            local item = example[itemShuffle[j]]
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
        -- Fill empty
        for j = #example, synthqa.NUM_GRID do
            result[{i, (j - 1) * numDim + 1}] = #synthqa.OBJECT + 1
            result[{i, (j - 1) * numDim + 2}] = #synthqa.COLOR + 1
            result[
                    {i, {(j - 1) * numDim + 3, (j - 1) * numDim + 6}}] = 
                    getCoord(value)
        end
    end
    return result
end

synthqa.dict = combine({
            synthqa.OBJECT, 
            synthqa.X, 
            synthqa.Y, 
            synthqa.COLOR, 
            synthqa.RELATION, 
            synthqa.WORDS,
            synthqa.numObj1
        })
synthqa.idict = imageqa.invertDict(synthqa.dict)

----------------------------------------------------------------------

function synthqa.prep(rawData)
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
    local labels = answerIds
    return data, labels
end


function synthqa.createModel(params)
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

    -- Input
    local input = nn.Identity()()

    -- Items to attend
    local itemRawDim = params.objectEmbedDim + params.colorEmbedDim + 4
    local items = nn.Narrow(2, 1, 54)(input)
    local itemsReshape = nn.Reshape(9, 6)(items)
    local catId = nn.Select(3, 1)(itemsReshape)
    local catIdReshape = nn.BatchReshape(1)(catId)
    local colorId = nn.Select(3, 2)(itemsReshape)
    local colorIdReshape = nn.BatchReshape(1)(colorId)
    local coord = nn.Narrow(3, 3, 4)(itemsReshape)
    local coordReshape = nn.BatchReshape(4)(coord)
    local catEmbed = nn.LookupTable(
        params.numObject, params.objectEmbedDim)(catIdReshape)
    local colorEmbed = nn.LookupTable(
        params.numColor, params.colorEmbedDim)(colorIdReshape)
    local itemsJoined = nn.JoinTable(2, 2)(
        {catEmbed, colorEmbed, coordReshape})
    local itemsJoinedReshape = nn.Reshape(9, itemRawDim)(itemsJoined)

    -- Word Embeddings
    local wordIds = nn.Narrow(2, 55, params.questionLength)(input)
    local wordEmbed = nn.LookupTable(
        #synthqa.idict, params.wordEmbedDim)(wordIds)
    local wordEmbedSeq = nn.SplitTable(2)(wordEmbed)
    local inputToEncoder = nn.gModule(
        {input}, {wordEmbedSeq, itemsJoinedReshape})

    -- Encoder LSTM
    local encoderCore = lstm.createUnit(params.wordEmbedDim, params.lstmDim)
    local encoder = nn.RNN(encoderCore, params.questionLength)

    -- Encoder state transfer
    local encoderOutput = nn.Identity()()
    local encoderOutputSel = nn.SelectTable(
        params.questionLength)(encoderOutput)
    local encoderStateTransfer = nn.gModule(
        {encoderOutput}, {encoderOutputSel})

    -- Decoder dummy input
    local decoderInputId = nn.Identity()()
    local decoderInputConst = nn.Constant(
        {params.decoderSteps, 1}, 0)(decoderInputId)
    local decoderInputSplit = nn.SplitTable(2)(decoderInputConst)
    local decoderInput = nn.gModule({decoderInputId}, {decoderInputSplit})

    -- Select items
    local itemsSelectInput = nn.Identity()()
    local itemsSelectTable = nn.SelectTable(2)(itemsSelectInput)
    local itemsSelect = nn.gModule({itemsSelectInput}, {itemsSelectTable})

    -- Decoder LSTM
    local decoderCore = lstm.createAttentionUnit(
        1, params.lstmDim, 9, itemRawDim)
    local decoder = nn.RNN(decoderCore, params.decoderSteps)

    -- Classify answer
    local decoderOutput = nn.Identity()()
    local decoderOutputSel = nn.SelectTable(
        params.decoderSteps)(decoderOutput)
    local decoderOutputState = nn.NarrowTable(
        2, params.lstmDim + 1, params.lstmDim)(decoderOutputSel)
    local classificationLayer = nn.Linear(
        params.lstmDim, params.vocabSize)(decoderOutputState)
    local classification = nn.gModule({decoderOutput}, {classificationLayer})

    -- Build entire model
    local all = nn.LazySequential()

    all:addModule('input', inputToEncoder)
    local encoderConcat = nn.ConcatTable()
    local encoderSeq = nn.Sequential()
    encoderSeq:add(encoder)
    encoderSeq:add(encoderStateTransfer)
    encoderConcat:add(decoderInput)
    encoderConcat:add(itemsSelect)
    encoderConcat:add(encoderSeq)

    all:addModule('encoderConcat', encoderConcat)
    all:addModule('decoder', decoder)
    all:addModule('answer', classification)
    all:setup()

    -- Expand LSTMs
    encoder:expand()
    decoder:expand()

    return all
end

----------------------------------------------------------------------
local N = 100
local rawData = synthqa.genHowManyObject(N)
local data, labels = synthqa.prep(rawData)
logger:logInfo(data:size())
logger:logInfo(labels:size())
data = {
    trainData = data[{{1, torch.floor(N / 2)}}],
    trainLabels = labels[{{1, torch.floor(N / 2)}}],
    testData = data[{{torch.floor(N / 2) + 1, N}}],
    testLabels = labels[{{torch.floor(N / 2) + 1, N}}]
}

local params = {
    objectEmbedDim = 2,
    colorEmbedDim = 2,
    questionLength = 3,
    wordEmbedDim = 10,
    lstmDim = 10,
    itemDim = 10,
    decoderSteps = 8,
    vocabSize = #synthqa.idict,
    numObject = #synthqa.OBJECT + 1,
    numColor = #synthqa.COLOR + 1
}
local model = synthqa.createModel(params)
print(model)
model.criterion = nn.MSECriterion()
model.decision = function(prediction)
    local score, idx = prediction:max(2)
    return idx
end

local optimConfig = {
    learningRate = 0.1,
    --learningRates = learningRates,
    momentum = 0.9,
    --gradientClip = utils.gradientClip(gradClipTable, model.sliceLayer)
}

local loopConfig = {
    numEpoch = 10000,
    trainBatchSize = 20,
    evalBatchSize = 1000,
    progressBar = false
}

local NNTrainer = require('nntrainer')
local optim = require('optim')
local optimizer = optim.sgd
local trainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
trainer:trainLoop(
    data.trainData, data.trainLabels, data.testData, data.testLabels)