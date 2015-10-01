local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local weights = require('weights')
local batch_reshape = require('batch_reshape')
local logger = require('logger')()
local dpnn = require('dpnn')
local memnn = {}

-------------------------------------------------------------------------------
function memnn.createMemoryUnitWithBinaryOutput(inputDim, numMemory, memoryDim)
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
    local memoryNextReshape = mynn.BatchReshape(
        numMemory * memoryDim)(memoryNext)
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

-------------------------------------------------------------------------------
function memnn.createMemoryUnitWithBinaryOutput2(
    inputDim, numMemory, memoryDim)
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

-------------------------------------------------------------------------------
function memnn.createMemoryUnitWithBinaryOutput3(
    inputDim, numMemory, memoryDim, hasQueryWrite, forget)
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
    local memoryPrev = mynn.BatchReshape(
        numMemory, memoryDim)(memoryPrevReshape)

    local inputExpand = nn.Replicate(numMemory, 2)(input)
    local inputExpandReshape = mynn.BatchReshape(inputDim)(inputExpand)
    local memoryPrevReshape2 = mynn.BatchReshape(memoryDim)(memoryPrev)
    local inputExpandLTRead = nn.Linear(
        inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevLTRead = nn.Linear(
        memoryDim, memoryDim)(memoryPrevReshape2)

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
        attWeightWrite.data.module.weight:uniform(
            -initRange / 2, initRange / 2)
        inputExpandLTWrite.data.module.weight:uniform(
            -initRange / 2, initRange / 2)
        memoryPrevLTWrite.data.module.weight:uniform(
            -initRange / 2, initRange / 2)
    end

    writeHead.data.module.weight:uniform(-initRange / 2, initRange / 2)
    return coreModule
end

-------------------------------------------------------------------------------
function memnn.createMemoryUnitWithBinaryOutput4(
    numMemory, memoryDim, hasQueryWrite, forget)
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
    local memoryPrev = mynn.BatchReshape(
        numMemory, memoryDim)(memoryPrevReshape)
    local inputExpand = nn.Replicate(numMemory, 2)(input)
    local inputExpandReshape = mynn.BatchReshape(inputDim)(inputExpand)
    local memoryPrevReshape2 = mynn.BatchReshape(memoryDim)(memoryPrev)
    local inputExpandLTRead = nn.Linear(
        inputDim, memoryDim)(inputExpandReshape)
    local memoryPrevLTRead = nn.Linear(
        memoryDim, memoryDim)(memoryPrevReshape2)

    local queryRead = nn.Tanh()(
        nn.CAddTable()({inputExpandLTRead, memoryPrevLTRead}))
    local queryReadReshape = mynn.BatchReshape(numMemory, memoryDim)(queryRead)
    local attWeightRead = mynn.Weights(memoryDim)(input)
    attWeightRead.data.module.name = 'attWeight'
    local attWeightReadReshape = mynn.BatchReshape(memoryDim, 1)(attWeightRead)
    local mmRead = nn.MM()({queryReadReshape, attWeightReadReshape})
    mmRead.data.module.name = 'mmRead'
    local queryReadResult = mynn.BatchReshape(numMemory)(mmRead)

    local inputLT = nn.Tanh()(nn.Linear(inputDim, numMemory)(input))
    local writeHead = nn.CSubTable()({inputLT, queryReadResult})
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
    return coreModule
end

-------------------------------------------------------------------------------
return memnn
