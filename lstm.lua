local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local weights = require('weights')
local batch_reshape = require('batch_reshape')
local logger = require('logger')()
local dpnn = require('dpnn')
local lstm = {}

----------------------------------------------------------------------
function lstm.createBinaryInputUnit(inputDim2, hiddenDim, initRange)
    if initRange == nil then
        initRange = 0.1
    end
    local inputDim = 1
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    
    local inputHidden = nn.Narrow(2, inputDim2 + 1, inputDim2)(input)
    local aggregate = nn.Linear(inputDim2, 1)(inputHidden)
    local sigmoid = nn.Sigmoid()(aggregate)
    local stochastic = nn.ReinforceBernoulli()(sigmoid)

    local joinState1 = nn.JoinTable(2, 2)({stochastic, statePrev})
    joinState1.data.module.name = 'joinState1'
    local joinState2 = nn.JoinTable(2, 2)({stochastic, hiddenPrev})
    joinState2.name = 'joinState2'
    local state1 = hiddenDim * 2 + inputDim
    local state2 = hiddenDim + inputDim
    local inGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local inGate = nn.Sigmoid()(inGateLin)
    local inTransformLin = nn.Linear(state2, hiddenDim)(joinState2)
    local inTransform = nn.Tanh()(inTransformLin)
    local forgetGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local forgetGate = nn.Sigmoid()(forgetGateLin)
    local outGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local outGate = nn.Sigmoid()(outGateLin)
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cellPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})
    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext})
    stateNext.data.module.name = 'stateNext'
    local stackModule = nn.gModule({input, statePrev}, {stateNext})
    stackModule.reinforceUnit = stochastic.data.module
    stackModule.moduleMap = {
        input = stochastic.data.module
    }

    aggregate.data.module.weight:uniform(-initRange / 2, initRange / 2)
    aggregate.data.module.bias:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)
    return stackModule
end


function lstm.createUnit(inputDim, hiddenDim, initRange)
    if initRange == nil then
        initRange = 0.1
    end
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    local joinState1 = nn.JoinTable(2, 2)({input, statePrev})
    joinState1.data.module.name = 'joinState1'
    local joinState2 = nn.JoinTable(2, 2)({input, hiddenPrev})
    joinState2.data.module.name = 'joinState2'
    local state1 = hiddenDim * 2 + inputDim
    local state2 = hiddenDim + inputDim
    local inGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local inGate = nn.Sigmoid()(inGateLin)
    local inTransformLin = nn.Linear(state2, hiddenDim)(joinState2)
    local inTransform = nn.Tanh()(inTransformLin)
    local forgetGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local forgetGate = nn.Sigmoid()(forgetGateLin)
    local outGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local outGate = nn.Sigmoid()(outGateLin)
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cellPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})
    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext})
    stateNext.data.module.name = 'stateNext'
    local stackModule = nn.gModule({input, statePrev}, {stateNext})
    stackModule.moduleMap = {
        hiddenNext = hiddenNext.data.module
    }

    inGateLin.data.module.name = 'inGateLin'
    inTransformLin.data.module.name = 'inTransformLin'
    forgetGateLin.data.module.name = 'forgetGateLin'
    outGateLin.data.module.name = 'outGateLin'
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)
    return stackModule
end

----------------------------------------------------------------------
function lstm.createAttentionUnitWithBinaryOutput(
                                    inputDim, 
                                    hiddenDim, 
                                    numItems, 
                                    itemDim, 
                                    initRange, 
                                    attentionMechanism,
                                    training)
    if initRange == nil then
        initRange = 0.1
    end
    -- Soft or hard attention
    if attentionMechanism == nil then
        attentionMechanism = 'soft'
    end
    if training == nil then
        training = false
    end
    local input = nn.Identity()()

    local statePrev = nn.Identity()()
    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    local binaryOutputPrev = nn.Narrow(2, hiddenDim * 2, 1)(statePrev)

    local itemsReshape = mynn.BatchReshape(itemDim)(items)
    itemsReshape.data.module.name = 'itemsReshape'
    local hiddenPrevLT = nn.Linear(hiddenDim, itemDim)(hiddenPrev)
    local hiddenPrevLTRepeat = nn.Replicate(numItems, 2)(hiddenPrevLT)

    local itemsLT = nn.Linear(itemDim, itemDim)(itemsReshape)
    local itemsLTReshape = mynn.BatchReshape(numItems, itemDim)(itemsLT)
    itemsLTReshape.data.module.name = 'itemsLTReshape'

    local query = nn.Tanh()(
        nn.CAddTable()({hiddenPrevLTRepeat, itemsLTReshape}))

    local attWeight = mynn.Weights(itemDim)(input)
    local attentionWeight = nn.Reshape(itemDim, 1)(attWeight)
    attentionWeight.data.module.name = 'attentionWeight'
    local mm1 = nn.MM()({query, attentionWeight})
    mm1.data.module.name = 'MM1'
    local attention = nn.Reshape(numItems)(mm1)
    attention.data.module.name = 'attention'
    local attentionNorm = nn.SoftMax()(attention)
    
    -- Build attention using either 'hard' or 'soft' attention
    local attention
    if attentionMechanism == 'hard' then
        attention = nn.ReinforceCategorical(training)(attentionNorm)
    elseif attentionMechanism == 'soft' then
        attention = attentionNorm
    else
        logger:logFatal(string.format(
            'unknown attention mechanism: %s', attentionMechanism))
    end

    local attentionReshape = nn.Reshape(1, numItems)(attention)
    local mm2 = nn.MM()({attentionReshape, items})
    mm2.data.module.name = 'MM2'
    local attendedItems = nn.Reshape(itemDim)(mm2)

    local joinState1 = nn.JoinTable(2, 2)(
        {input, attendedItems, attention, statePrev})
    joinState1.data.module.name = 'attJoinState1'
    local joinState2 = nn.JoinTable(2, 2)(
        {input, attendedItems, attention, hiddenPrev, binaryOutputPrev})
    joinState2.data.module.name = 'attJoinState2'
    local state1 = hiddenDim * 2 + inputDim + itemDim + numItems + 1
    local state2 = hiddenDim + inputDim + itemDim + numItems + 1
    local inGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local inGate = nn.Sigmoid()(inGateLin)
    local inTransformLin = nn.Linear(state2, hiddenDim)(joinState2)
    local inTransform = nn.Tanh()(inTransformLin)
    local forgetGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local forgetGate = nn.Sigmoid()(forgetGateLin)
    local outGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local outGate = nn.Sigmoid()(outGateLin)
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cellPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})
    local attentionAndHidden = nn.JoinTable(2)({attendedItems, hiddenNext})
    -- local aggregate = nn.Linear(hiddenDim, 1)(hiddenNext)
    local aggregate = nn.Linear(hiddenDim + itemDim, 1)(attentionAndHidden)
    local binarySoftOutput = nn.Sigmoid()(aggregate)
    local binaryOutput = nn.ReinforceBernoulli()(binarySoftOutput)

    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext, binaryOutput})
    stateNext.name = 'attStateNext'
    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})

    coreModule.moduleMap = {
        softAttention = attentionNorm.data.module,
        attention = attention.data.module,
        binarySoftOutput = binarySoftOutput.data.module,
        binaryOutput = binaryOutput.data.module
    }
    
    inGateLin.data.module.name = 'attInGateLin'
    inTransformLin.data.module.name = 'attInTransformLin'
    forgetGateLin.data.module.name = 'attForgetGateLin'
    outGateLin.data.module.name = 'attOutGateLin'
    aggregate.data.module.name = 'attAggregate'

    aggregate.data.module.weight:uniform(-initRange / 2, initRange / 2)
    attWeight.data.module.weight:uniform(-initRange / 2, initRange / 2)
    itemsLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    hiddenPrevLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)
    return coreModule
end

function lstm.createAttentionUnit(
                                    inputDim, 
                                    hiddenDim, 
                                    numItems, 
                                    itemDim, 
                                    initRange, 
                                    attentionMechanism,
                                    training)
    if initRange == nil then
        initRange = 0.1
    end
    -- Soft or hard attention
    if attentionMechanism == nil then
        attentionMechanism = 'soft'
    end
    if training == nil then
        training = false
    end
    local input = nn.Identity()()

    local statePrev = nn.Identity()()
    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)

    local itemsReshape = mynn.BatchReshape(itemDim)(items)
    itemsReshape.data.module.name = 'itemsReshape'
    local hiddenPrevLT = nn.Linear(hiddenDim, itemDim)(hiddenPrev)
    local hiddenPrevLTRepeat = nn.Replicate(numItems, 2)(hiddenPrevLT)

    local itemsLT = nn.Linear(itemDim, itemDim)(itemsReshape)
    local itemsLTReshape = mynn.BatchReshape(numItems, itemDim)(itemsLT)
    itemsLTReshape.data.module.name = 'itemsLTReshape'

    local query = nn.Tanh()(
        nn.CAddTable()({hiddenPrevLTRepeat, itemsLTReshape}))

    local attWeight = mynn.Weights(itemDim)(input)
    local attentionWeight = nn.Reshape(itemDim, 1)(attWeight)
    attentionWeight.data.module.name = 'attentionWeight'
    local mm1 = nn.MM()({query, attentionWeight})
    mm1.data.module.name = 'MM1'
    local attention = nn.Reshape(numItems)(mm1)
    attention.data.module.name = 'attention'
    local attentionNorm = nn.SoftMax()(attention)
    
    -- Build attention using either 'hard' or 'soft' attention
    local attention
    if attentionMechanism == 'hard' then
        attention = nn.ReinforceCategorical(training)(attentionNorm)
    elseif attentionMechanism == 'soft' then
        attention = attentionNorm
    else
        logger:logFatal(string.format(
            'unknown attention mechanism: %s', attentionMechanism))
    end

    local attentionReshape = nn.Reshape(1, numItems)(attention)
    local mm2 = nn.MM()({attentionReshape, items})
    mm2.data.module.name = 'MM2'
    local attendedItems = nn.Reshape(itemDim)(mm2)

    local joinState1 = nn.JoinTable(2)({input, attendedItems, cellPrev, hiddenPrev})
    joinState1.data.module.name = 'attJoinState1'
    local joinState2 = nn.JoinTable(2)({input, attendedItems, hiddenPrev})
    joinState2.data.module.name = 'attJoinState2'
    local state1 = hiddenDim * 2 + inputDim + itemDim
    local state2 = hiddenDim + inputDim + itemDim
    local inGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local inGate = nn.Sigmoid()(inGateLin)
    local inTransformLin = nn.Linear(state2, hiddenDim)(joinState2)
    local inTransform = nn.Tanh()(inTransformLin)
    local forgetGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local forgetGate = nn.Sigmoid()(forgetGateLin)
    local outGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local outGate = nn.Sigmoid()(outGateLin)
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cellPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})
    local stateNext = nn.JoinTable(2)({cellNext, hiddenNext, attendedItems, attentionNorm})
    stateNext.data.module.name = 'attStateNext'
    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})
    coreModule.reinforceUnit = attention.data.module
    coreModule.moduleMap = {
        softAttention = attentionNorm.data.module,
        attention = attention.data.module
    }
    
    attWeight.data.module.weight:uniform(-initRange / 2, initRange / 2)
    itemsLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    hiddenPrevLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)
    return coreModule
end


----------------------------------------------------------------------
function lstm.createAttentionUnitDebug(
                                    inputDim, 
                                    hiddenDim, 
                                    numItems, 
                                    itemDim, 
                                    initRange, 
                                    attentionMechanism,
                                    training)
    if initRange == nil then
        initRange = 0.1
    end
    -- Soft or hard attention
    if attentionMechanism == nil then
        attentionMechanism = 'soft'
    end
    if training == nil then
        training = false
    end
    local input = nn.Identity()()

    local statePrev = nn.Identity()()
    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)

    local itemsReshape = mynn.BatchReshape(itemDim)(items)
    itemsReshape.data.module.name = 'itemsReshape'
    local hiddenPrevLT = nn.Linear(hiddenDim, itemDim)(hiddenPrev)
    local hiddenPrevLTRepeat = nn.Replicate(numItems, 2)(hiddenPrevLT)

    local itemsLT = nn.Linear(itemDim, itemDim)(itemsReshape)
    local itemsLTReshape = mynn.BatchReshape(numItems, itemDim)(itemsLT)
    itemsLTReshape.data.module.name = 'itemsLTReshape'

    local query = nn.Tanh()(
        nn.CAddTable()({hiddenPrevLTRepeat, itemsLTReshape}))

    local attWeight = mynn.Weights(itemDim)(input)
    local attentionWeight = nn.Reshape(itemDim, 1)(attWeight)
    attentionWeight.data.module.name = 'attentionWeight'
    local mm1 = nn.MM()({query, attentionWeight})
    mm1.data.module.name = 'MM1'
    local attention = nn.Reshape(numItems)(mm1)
    attention.data.module.name = 'attention'
    local attentionNorm = nn.SoftMax()(attention)

    local attentionReshape = nn.Reshape(1, numItems)(attention)
    local mm2 = nn.MM()({attentionReshape, items})
    mm2.data.module.name = 'MM2'
    local attendedItems = nn.Reshape(itemDim)(mm2)

    local joinState1 = nn.JoinTable(2, 2)({input, attendedItems, statePrev})
    joinState1.name = 'attJoinState1'
    local joinState2 = nn.JoinTable(2, 2)({input, attendedItems, hiddenPrev})
    joinState2.name = 'attJoinState2'
    local state1 = hiddenDim * 2 + inputDim + itemDim + numItems
    local state2 = hiddenDim + inputDim + itemDim
    local inGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local inGate = nn.Sigmoid()(inGateLin)
    local inTransformLin = nn.Linear(state2, hiddenDim)(joinState2)
    local inTransform = nn.Tanh()(inTransformLin)
    local forgetGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local forgetGate = nn.Sigmoid()(forgetGateLin)
    local outGateLin = nn.Linear(state1, hiddenDim)(joinState1)
    local outGate = nn.Sigmoid()(outGateLin)
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cellPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})
    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext, attentionNorm})
    stateNext.name = 'attStateNext'
    local coreModule = nn.gModule({input, statePrev, items}, {stateNext})
    coreModule.moduleMap = {
        softAttention = attentionNorm.data.module
        -- attention = attention.data.module
    }
    
    attWeight.data.module.weight:uniform(-initRange / 2, initRange / 2)
    itemsLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    hiddenPrevLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLin.data.module.bias:fill(1)
    inTransformLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inTransformLin.data.module.bias:fill(0)
    forgetGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    forgetGateLin.data.module.bias:fill(1)
    outGateLin.data.module.weight:uniform(-initRange / 2, initRange / 2)
    outGateLin.data.module.bias:fill(1)
    return coreModule
end

function lstm.createMemoryUnitWithBinaryOutput(inputDim, numMemory, memoryDim)
    -- Input to this module is
    -- Input at timestep t
    -- Memory content
    local initRange = 0.1
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local memoryPrevReshape = nn.Narrow(2, 1, numMemory * memoryDim)(statePrev)
    local memoryPrev = mynn.BatchReshape(
        numMemory, memoryDim)(memoryPrevReshape)

    local inputExpand = nn.Replicate(numMemory, 2)(input)
    local inputExpandReshape = mynn.BatchReshape(inputDim)(inputExpand)
    local inputExpandLT = nn.Linear(inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevReshape2 = mynn.BatchReshape(memoryDim)(memoryPrev)
    local memoryPrevLT = nn.Linear(memoryDim, memoryDim)(memoryPrevReshape2)

    local query = nn.Tanh()(
        nn.CAddTable()({inputExpandLT, memoryPrevLT}))
    local queryReshape = mynn.BatchReshape(numMemory, memoryDim)(query)
    local attWeight = mynn.Weights(memoryDim)(input)
    attWeight.data.module.name = 'attWeight'
    local attWeightReshape = mynn.BatchReshape(memoryDim, 1)(attWeight)
    local mm1 = nn.MM()({queryReshape, attWeightReshape})
    mm1.data.module.name = 'MM1'
    local queryResult = mynn.BatchReshape(numMemory)(mm1)
    local queryOutputMap = nn.Linear(numMemory, 1)(queryResult)
    local queryOutput = nn.Sigmoid()(queryOutputMap)

    local writeHead = nn.Linear(inputDim, numMemory)(input)
    local writeHeadNorm = nn.SoftMax()(writeHead)
    local writeHeadExp = nn.Replicate(memoryDim, 3)(writeHeadNorm)
    local addMemory = nn.CMulTable()({inputExpandLT, writeHeadExp})
    local memoryNext = nn.CAddTable()({memoryPrev, addMemory})
    local memoryNextReshape = mynn.BatchReshape(numMemory * memoryDim)(memoryNext)
    local stateNext = nn.JoinTable(2)({memoryNextReshape, queryOutput})

    local coreModule = nn.gModule({input, statePrev}, {stateNext})

    coreModule.moduleMap = {
        queryOutput = queryOutput.data.module,
        writeHeadNorm = writeHeadNorm.data.module
    }

    attWeight.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inputExpandLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    memoryPrevLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    writeHead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    queryOutputMap.data.module.weight:uniform(-initRange / 2, initRange / 2)

    return coreModule
end

----------------------------------------------------------------------
function lstm.createMemoryUnitWithBinaryOutput2(inputDim, numMemory, memoryDim)
    -- Input to this module is
    -- Input at timestep t
    -- Memory content
    local initRange = 0.1
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local memoryPrevReshape = nn.Narrow(2, 1, numMemory * memoryDim)(statePrev)
    -- local writeHeadNormPrev = nn.Narrow(2, numMemory * memoryDim + numMemory + 1, numMemory)(statePrev)
    local memoryPrev = mynn.BatchReshape(
        numMemory, memoryDim)(memoryPrevReshape)

    local inputExpand = nn.Replicate(numMemory, 2)(input)
    local inputExpandReshape = mynn.BatchReshape(inputDim)(inputExpand)
    local memoryPrevReshape2 = mynn.BatchReshape(memoryDim)(memoryPrev)
    local inputExpandLTRead = nn.Linear(inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevLTRead = nn.Linear(memoryDim, memoryDim)(memoryPrevReshape2)

    local queryRead = nn.Tanh()(
        nn.CAddTable()({inputExpandLTRead, memoryPrevLTRead}))
    local queryReadReshape = mynn.BatchReshape(numMemory, memoryDim)(queryRead)
    local attWeightRead = mynn.Weights(memoryDim)(input)
    attWeightRead.data.module.name = 'attWeight'
    local attWeightReadReshape = mynn.BatchReshape(memoryDim, 1)(attWeightRead)
    local mmRead = nn.MM()({queryReadReshape, attWeightReadReshape})
    mmRead.data.module.name = 'mmRead'
    local queryReadResult = mynn.BatchReshape(numMemory)(mmRead)

    local inputExpandLTWrite = nn.Linear(inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevLTWrite = nn.Linear(memoryDim, memoryDim)(memoryPrevReshape2)
    local queryWrite = nn.Tanh()(
        nn.CAddTable()({inputExpandLTWrite, memoryPrevLTWrite}))
    local queryWriteReshape = mynn.BatchReshape(numMemory, memoryDim)(queryWrite)
    local attWeightWrite = mynn.Weights(memoryDim)(input)
    attWeightWrite.data.module.name = 'attWeightWrite'
    local attWeightWriteReshape = mynn.BatchReshape(memoryDim, 1)(attWeightWrite)
    local mmWrite = nn.MM()({queryWriteReshape, attWeightWriteReshape})
    mmWrite.data.module.name = 'mmWrite'
    local queryWriteResult = mynn.BatchReshape(numMemory)(mmWrite)

    local inputWriteHead = nn.Linear(inputDim, numMemory)(input)
    local writeHead = nn.CAddTable()({queryWriteResult, inputWriteHead})
    -- local writeHead2 = nn.CSubTable()({writeHead, writehead2})
    local writeHeadNorm = nn.SoftMax()(writeHead)
    local writeHeadNormExp = nn.Replicate(memoryDim, 3)(writeHeadNorm)
    local addMemory = nn.CMulTable()({inputExpandLTWrite, writeHeadNormExp})

    local memoryNext = nn.CAddTable()({memoryPrevReshape, addMemory})
    local memoryNextReshape = mynn.BatchReshape(numMemory * memoryDim)(memoryNext)
    local stateNext = nn.JoinTable(2)({memoryNextReshape, queryReadResult})

    local coreModule = nn.gModule({input, statePrev}, {stateNext})

    coreModule.moduleMap = {
        queryReadResult = queryReadResult.data.module,
        writeHeadNorm = writeHeadNorm.data.module
    }

    attWeightRead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inputExpandLTRead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    memoryPrevLTRead.data.module.weight:uniform(-initRange / 2, initRange / 2)

    attWeightWrite.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inputExpandLTWrite.data.module.weight:uniform(-initRange / 2, initRange / 2)
    memoryPrevLTWrite.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inputWriteHead.data.module.weight:uniform(-initRange / 2, initRange / 2)

    return coreModule
end

----------------------------------------------------------------------
function lstm.createMemoryUnitWithBinaryOutput3(inputDim, numMemory, memoryDim, hasQueryWrite, forget)
    if hasQueryWrite == nil then
        hasQueryWrite = false
    end
    if forget == nil then
        forget = false
    end
    local initRange = 0.1
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local memoryPrevReshape = nn.Narrow(2, 1, numMemory * memoryDim)(statePrev)
    -- local writeHeadNormPrev
    -- if hasQueryWrite then
    --     writeHeadNormPrev = nn.Narrow(
    --         2, numMemory * memoryDim + memoryDim + 1, numMemory)(statePrev)
    -- end
    local memoryPrev = mynn.BatchReshape(
        numMemory, memoryDim)(memoryPrevReshape)

    local inputExpand = nn.Replicate(numMemory, 2)(input)
    local inputExpandReshape = mynn.BatchReshape(inputDim)(inputExpand)
    local memoryPrevReshape2 = mynn.BatchReshape(memoryDim)(memoryPrev)
    local inputExpandLTRead = nn.Linear(inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevLTRead = nn.Linear(memoryDim, memoryDim)(memoryPrevReshape2)

    local queryRead = nn.Tanh()(
        nn.CAddTable()({inputExpandLTRead, memoryPrevLTRead}))
    local queryReadReshape = mynn.BatchReshape(numMemory, memoryDim)(queryRead)
    local attWeightRead = mynn.Weights(memoryDim)(input)
    attWeightRead.data.module.name = 'attWeight'
    local attWeightReadReshape = mynn.BatchReshape(memoryDim, 1)(attWeightRead)
    local mmRead = nn.MM()({queryReadReshape, attWeightReadReshape})
    mmRead.data.module.name = 'mmRead'
    local queryReadResult = mynn.BatchReshape(numMemory)(mmRead)

    local inputExpandLTWrite, memoryPrevLTWrite, queryWrite, queryWriteReshape
    local attWeightWrite, attWeightWriteReshape, mmWrite, queryWriteResult
    local writeHeadJoin
    local writeHead
    if hasQueryWrite then
        inputExpandLTWrite = nn.Linear(
            inputDim, memoryDim)(inputExpandReshape)
        memoryPrevLTWrite = nn.Linear(
            memoryDim, memoryDim)(memoryPrevReshape2)
        queryWrite = nn.Tanh()(
            nn.CAddTable()({inputExpandLTWrite, memoryPrevLTWrite}))
        queryWriteReshape = mynn.BatchReshape(
            numMemory, memoryDim)(queryWrite)
        attWeightWrite = mynn.Weights(memoryDim)(input)
        attWeightWrite.data.module.name = 'attWeightWrite'
        attWeightWriteReshape = mynn.BatchReshape(
            memoryDim, 1)(attWeightWrite)
        mmWrite = nn.MM()({queryWriteReshape, attWeightWriteReshape})
        mmWrite.data.module.name = 'mmWrite'
        queryWriteResult = mynn.BatchReshape(numMemory)(mmWrite)
        writeHeadJoin = nn.JoinTable(2)(
            {queryWriteResult, input})
        writeHead = nn.Linear(
            numMemory + inputDim, numMemory)(writeHeadJoin)
    else
        writeHead = nn.Linear(inputDim, numMemory)(input)
    end
    local writeHeadNorm = nn.SoftMax()(writeHead)
    local writeHeadNormExp = nn.Replicate(memoryDim, 3)(writeHeadNorm)
    local addMemory

    if hasQueryWrite then
        addMemory = nn.CMulTable()({inputExpandLTWrite, writeHeadNormExp})
    else
        addMemory = nn.CMulTable()({inputExpandLTRead, writeHeadNormExp})
    end
    
    local memoryNext
    if forget then
        local ones = mynn.Constant({numMemory, memoryDim}, 1)(input)
        local forgetGate = nn.CSubTable()({ones, writeHeadNormExp})
        local forgetMemory = nn.CMulTable()({memoryPrevReshape, forgetGate})
        memoryNext = nn.CAddTable()({forgetMemory, addMemory})
    else
        memoryNext = nn.CAddTable()({memoryPrevReshape, addMemory})
    end

    local memoryNextReshape = mynn.BatchReshape(
        numMemory * memoryDim)(memoryNext)
    local stateNext = nn.JoinTable(2)(
        {memoryNextReshape, queryReadResult, writeHeadNorm})
    local coreModule = nn.gModule({input, statePrev}, {stateNext})

    coreModule.moduleMap = {
        -- queryReadOutputMap = queryReadOutputMap.data.module,
        queryReadResult = queryReadResult.data.module,
        writeHeadNorm = writeHeadNorm.data.module
    }

    attWeightRead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inputExpandLTRead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    memoryPrevLTRead.data.module.weight:uniform(-initRange / 2, initRange / 2)

    if hasQueryWrite then
        attWeightWrite.data.module.weight:uniform(-initRange / 2, initRange / 2)
        inputExpandLTWrite.data.module.weight:uniform(-initRange / 2, initRange / 2)
        memoryPrevLTWrite.data.module.weight:uniform(-initRange / 2, initRange / 2)
    end
    
    writeHead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    return coreModule
end
----------------------------------------------------------------------


----------------------------------------------------------------------
function lstm.createMemoryUnitWithBinaryOutput4(numMemory, memoryDim, hasQueryWrite, forget)
    if hasQueryWrite == nil then
        hasQueryWrite = false
    end
    if forget == nil then
        forget = false
    end
    local initRange = 0.1
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local memoryPrevReshape = nn.Narrow(2, 1, numMemory * memoryDim)(statePrev)
    -- local writeHeadNormPrev
    -- if hasQueryWrite then
    --     writeHeadNormPrev = nn.Narrow(
    --         2, numMemory * memoryDim + memoryDim + 1, numMemory)(statePrev)
    -- end
    local memoryPrev = mynn.BatchReshape(
        numMemory, memoryDim)(memoryPrevReshape)
    local inputExpand = nn.Replicate(numMemory, 2)(input)
    local inputExpandReshape = mynn.BatchReshape(inputDim)(inputExpand)
    local memoryPrevReshape2 = mynn.BatchReshape(memoryDim)(memoryPrev)
    local inputExpandLTRead = nn.Linear(inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevLTRead = nn.Linear(memoryDim, memoryDim)(memoryPrevReshape2)

    local queryRead = nn.Tanh()(
        nn.CAddTable()({inputExpandLTRead, memoryPrevLTRead}))
    local queryReadReshape = mynn.BatchReshape(numMemory, memoryDim)(queryRead)
    local attWeightRead = mynn.Weights(memoryDim)(input)
    attWeightRead.data.module.name = 'attWeight'
    local attWeightReadReshape = mynn.BatchReshape(memoryDim, 1)(attWeightRead)
    local mmRead = nn.MM()({queryReadReshape, attWeightReadReshape})
    mmRead.data.module.name = 'mmRead'
    local queryReadResult = mynn.BatchReshape(numMemory)(mmRead)

    -- local join = nn.JoinTable(2)({input, queryReadResult})
    -- local writeHead = nn.Linear(inputDim + numMemory, numMemory)(join)
    local inputLT = nn.Tanh()(nn.Linear(inputDim, numMemory)(input))
    local writeHead = nn.CSubTable()({inputLT, queryReadResult})
    -- local writeHead= 
    local writeHeadNorm = nn.SoftMax()(writeHead)
    local writeHeadNormExp = nn.Replicate(memoryDim, 3)(writeHeadNorm)
    local addMemory = nn.CMulTable()({inputExpand, writeHeadNormExp})
    
    local memoryNext
    if forget then
        local ones = mynn.Constant({numMemory, memoryDim}, 1)(input)
        local forgetGate = nn.CSubTable()({ones, writeHeadNormExp})
        local forgetMemory = nn.CMulTable()({memoryPrev, forgetGate})
        memoryNext = nn.CAddTable()({forgetMemory, addMemory})
    else
        memoryNext = nn.CAddTable()({memoryPrev, addMemory})
    end

    local memoryNextReshape = mynn.BatchReshape(
        numMemory * memoryDim)(memoryNext)
    local stateNext = nn.JoinTable(2)(
        {memoryNextReshape, queryReadResult, writeHeadNorm})
    local coreModule = nn.gModule({input, statePrev}, {stateNext})

    coreModule.moduleMap = {
        queryReadResult = queryReadResult.data.module,
        writeHeadNorm = writeHeadNorm.data.module
    }
    
    -- writeHead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    -- inputLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    return coreModule
end
----------------------------------------------------------------------
return lstm
