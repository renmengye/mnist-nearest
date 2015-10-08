local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local constant = require('constant')
local logger = require('logger')()
local gru = {}

function gru.createUnit(inputDim, hiddenDim, initRange, inGateBiasInit)
    if initRange == nil then
        initRange = 0.1
    end
    if inGateBiasInit == nil then
        inGateBiasInit = 1.0
    end
    local input = nn.Identity()()
    local hiddenPrev = nn.Identity()()

    local joinState = nn.JoinTable(2)({input, hiddenPrev})
    local joinStateDim = inputDim + hiddenDim
    local inGateLT = nn.Linear(joinStateDim, hiddenDim)(joinState)
    local inGate = nn.Sigmoid()(inGateLT)
    local transformLT = nn.Linear(joinStateDim, hiddenDim)(joinState)
    local transform = nn.Tanh()(transformLT)
    local ones = mynn.Constant(hiddenDim, 1)(input)
    local carryGate = nn.CSubTable()({ones, inGate})
    local hiddenNext = nn.CAddTable()({
        nn.CMulTable()({inGate, transform}),
        nn.CMulTable()({carryGate, hiddenPrev})
    })
    local coreUnit = nn.gModule({input, hiddenPrev}, {hiddenNext})
    coreUnit.moduleMap = {
        inGateLT = inGateLT.data.module,
        inGate = inGate.data.module,
        transformLT = transformLT.data.module,
        transform = transform.data.module,
        carryGate = carryGate.data.module
    }
    joinState.data.module.name = 'gru.joinState'
    inGateLT.data.module.name = 'gru.inGateLT'
    transformLT.data.module.name = 'gru.transformLT'
    inGateLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    inGateLT.data.module.bias:fill(inGateBiasInit)
    transformLT.data.module.weight:uniform(-initRange / 2, initRange / 2)
    transformLT.data.module.bias:fill(0)

    return coreUnit
end

return gru
