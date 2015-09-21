local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local weights = require('weights')
local batch_reshape = require('batch_reshape')
local logger = require('logger')()
local lstm = {}

----------------------------------------------------------------------
function lstm.createUnit(inputDim, hiddenDim, initRange)
    if initRange == nil then
        initRange = 0.1
    end
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)
    local joinState1 = nn.JoinTable(2, 2)({input, statePrev})
    local joinState2 = nn.JoinTable(2, 2)({input, hiddenPrev})
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
    local stackModule = nn.gModule({input, statePrev}, {stateNext})

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

function lstm.createAttentionUnit(
    inputDim, hiddenDim, numItems, itemDim, initRange)
    if initRange == nil then
        initRange = 0.1
    end
    local input = nn.Identity()()

    -- batch x numItems x itemDim
    local items = nn.Identity()()
    local statePrev = nn.Identity()()
    local cellPrev = nn.Narrow(2, 1, hiddenDim)(statePrev)
    local hiddenPrev = nn.Narrow(2, hiddenDim + 1, hiddenDim)(statePrev)

    local itemsReshape = nn.BatchReshape(itemDim)(items)
    local hiddenPrevLT = nn.Linear(hiddenDim, itemDim)(hiddenPrev)
    local hiddenPrevLTReshape = nn.Reshape(1, itemDim)(hiddenPrevLT)
    local hiddenPrevLTRepeat = nn.Replicate(numItems, 2)(hiddenPrevLTReshape)

    local itemsLT = nn.Linear(itemDim, itemDim)(itemsReshape)
    local itemsLTReshape = nn.BatchReshape(numItems, itemDim)(itemsLT)

    local query = nn.Tanh()(
        nn.CAddTable()({hiddenPrevLTRepeat, itemsLTReshape}))

    local attenionWeight = nn.Reshape(itemDim, 1)(nn.Weights(itemDim)(input))
    local attention = nn.Reshape(itemDim)(nn.MM()({query, attentionWeight}))
    local attentionNorm = nn.SoftMax()(attention)
    local attentionReshape = nn.Reshape(1, itemDim)(attentionNorm)
    local attendedItems = nn.Reshape(itemDim)(nn.MM()({attentionReshape, items}))

    local joinState1 = nn.JoinTable(2, 2)({input, attendedItems, statePrev})
    local joinState2 = nn.JoinTable(2, 2)({input, attendedItems, hiddenPrev})
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
    local stateNext = nn.JoinTable(2, 2)({cellNext, hiddenNext})
    local stackModule = nn.gModule({input, items, statePrev}, {stateNext})

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
return lstm
