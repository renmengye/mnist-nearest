local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local weights = require('weights')
local constant = require('constant')
local batch_reshape = require('batch_reshape')
local logger = require('logger')()
local dpnn = require('dpnn')
local rnn = require('rnn')
local one_hot = require('one_hot')
local expand_gmodule = require('expand_gmodule')
local lstm = {}

-------------------------------------------------------------------------------
function lstm.createLayer(inputDim, hiddenDim, timespan)
    local input = nn.Identity()()
    local inputSeq = nn.SelectTable(1)(input)
    local state0 = nn.SelectTable(2)(input)
    local core = lstm.createUnit(inputDim, hiddenDim)
    local rnn = nn.RNN(core, timespan)({input, constState})
    local layer = nn.ExpandGModule({input}, {rnn})
    layer:addModule('rnn', rnn)
    return layer
end

-------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------
return lstm
