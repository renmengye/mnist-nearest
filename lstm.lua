local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local rnn = require('rnn')
local logger = require('logger')()
local lstm = {}

----------------------------------------------------------------------
function lstm.createUnit(x, numInput, numHidden, cPrev, hPrev)
    -- * Create a LSTM unit, given some input layers.
    -- * This extends the layers, but itself is not a standalone module.
    -- * Use lstm.createStack to create input and output layers.
    local i2h = nn.Linear(numInput, 4 * numHidden)(x)
    local h2h = nn.Linear(numHidden, 4 * numHidden)(hPrev)
    local gates = nn.CAddTable()({i2h, h2h})
    local reshapedGates = nn.Reshape(4, numHidden)(gates)
    local slicedGates = nn.SplitTable(2)(reshapedGates)
    local inGate = nn.Sigmoid()(nn.SelectTable(1)(slicedGates))
    local inTransform = nn.Tanh()(nn.SelectTable(2)(slicedGates))
    local forgetGate = nn.Sigmoid()(nn.SelectTable(3)(slicedGates))
    local outGate = nn.Sigmoid()(nn.SelectTable(4)(slicedGates))
    local cellNext = nn.CAddTable()({
        nn.CMulTable()({forgetGate, cPrev}),
        nn.CMulTable()({inGate, inTransform})
    })
    local hiddenNext = nn.CMulTable()({outGate, nn.Tanh()(cellNext)})

    return cellNext, hiddenNext
end

function lstm.createStack(numInput, numHidden, numLayers)
    if numLayers == 0 then
        numLayers = 1
    end
    -- * Create a LSTM unit stack, including the input and ouput interface.
    -- * Eventually you will maintain T stacks as an RNN.
    -- * Each stack is a separate module, taking input of the previous state 
    -- variable and input variable.
    -- * It is up to you how the connections are wired accross timespan (e.g. 
    -- inferring generative RNN can be different).
    local input = nn.Identity()()
    local statePrev = nn.Identity()()
    local stateNext = {}
    local stateSplit = {statePrev:split(2 * numLayers)}
    for n = 1, numLayers do
        local cellPrev = stateSplit[2 * n - 1]
        local hiddenPrev = stateSplit[2 * n]
        local cellNext, hiddenNext = lstm.createUnit(
            input, numInput, numHidden, cellPrev, hiddenPrev)
        table.insert(stateNext, cellNext)
        table.insert(stateNext, hiddenNext)
    end
    local stackModule = nn.gModule({input}, stateNext)
    return stackModule
end

function lstm.forward(model)
    -- For recurrent networks model is a list of recurrent units.
    -- Each unit is accepting two inputs, state variable and input variable.

end

-- write a good model weights serializer: a dictionary!!
-- flatten weights doesn't really make sense other than optimization
-- for example conv net... flatten weights is really architecture dependent.



local MyModelClass, parent = torch.class('MyModel','nn.Module')

function MyModel:__init(numInput, numHidden, numLayers, timespan)
    self.numInput = numInput
    self.numHidden = numHidden
    self.numLayers = numLayers
    self.timespan = timespan
    local modelInput = nn.Identity()()
    local splitTable = nn.SplitTable(2)(modelInput)
    self.inputNode = nn.gModule({modelInput}, {splitTable})
    logger:logInfo(self.inputNode)
    local coreStack = lstm.createStack(numInput, numHidden, numLayers)
    self.rnn.replicas = rnn.cloneModule(coreStack, timespan)
    local lstmOutput = nn.Identity()()
    local linear = nn.Linear(1)(lstmOutput)
    local sigmoid = nn.Sigmoid()(linear)
    self.aggregator = nn.gModule({lstmOutput}, {sigmoid})
end

function MyModel:forward(x)
    xSplit = self.inputNode:forward(x)
    local state0 = {}
    state0[1] = torch.Tensor(self.numHidden):zero()
    state0[2] = torch.Tensor(self.numHidden):zero()
    local output = rnn.forward(self.rnn, xSplit, state0)
    return self.aggregator:forward(output[#output][2])
end

function MyModel:backward(x, dl_dy)
    local dl_drnn = {}
    for i = 1, model.rnn.timespan - 1 do
        dl_drnn[1] = torch.Tensor(self.rnn.numHidden):zero()
        dl_drnn[2] = torch.Tensor(self.rnn.numHidden):zero()
    end
    dl_drnn[self.rnn.timespan] = self.aggregator:backward(x, dl_dy)
    local dl_dxSplit, dl_ds0 = rnn.backward(self.rnn, x, dl_drnn)
    local dl_dx = self.inputNode:backward(x, dl_dxSplit)
    return dl_dx
end

function MyModel:parameters()
    local parameters = {}
    parameters[1] = self.rnn.replicas[1]:parameters()
    parameters[2] = self.aggregator:parameters()
    return parameters
end

function MyModel:training()
end

function MyModel:evaluate()
end

----------------------------------------------------------------------
function createModel(timespan)
    -- local model = {}
    -- model.rnn = {}
    -- local model.rnn.numInput = 1
    -- local model.rnn.numHidden = 5
    -- local model.rnn.numLayers = 1
    -- local model.rnn.timespan = 8
    -- local coreStack = lstm.createStack(
    --     model.rnn.numInput, model.rnn.numHidden, model.rnn.numLayers)
    -- model.rnn.replicas = rnn.cloneModule(coreStack, model.rnn.timespan)
    -- local lstmOutput = nn.Identity()()
    -- local linear = nn.Linear(1)(lstmOutput)
    -- model.aggregator = nn.gModule({lstmOutput}, {linear})

    -- local function forward(model, x)
    --     local state0 = {}
    --     state0[1] = torch.Tensor(model.rnn.numHidden):zero()
    --     state0[2] = torch.Tensor(model.rnn.numHidden):zero()
    --     local output = rnn.forward(model.rnn, x, state0)
    --     return model.aggregator:forward(output[#output][2])
    -- end
    
    -- local function backward(model, x, dl_dy)
    --     local dl_drnn = {}
    --     for i = 1, model.rnn.timespan - 1 do
    --         dl_drnn[1] = torch.Tensor(model.rnn.numHidden):zero()
    --         dl_drnn[2] = torch.Tensor(model.rnn.numHidden):zero()
    --     local dl_drnn[model.rnn.timespan] = model.aggregator:backward(x, dl_dy)
    --     local dl_dx, dl_ds0 = rnn.backward(model.rnn, x, dl_drnn)
    --     return dl_dx
    -- end

    -- local function getParameters(model)
    -- end

    -- return model, forward, backward

    return MyModel(1, 5, 1, timespan)
end

-- Generate train data
local N = 2000
local T = 8
local data = torch.Tensor(N, T, 1)
local label = torch.Tensor(N, 1)
for i = 1, N do
    for t = 1, T do
        data[i][t][1] = torch.floor(torch.uniform() * 2)
    end

end
local loopConfig = {
    numEpoch = 1000,
    trainBatchSize = 20,
    evalBatchSize = 1000
}

local optimConfig = {
    learningRate = 0.01,
    momentum = 0.9
}

local NNTrainer = require('nntrainer')
local optim = require('optim')
local optimizer = optim.sgd
local model = createModel(T)
model.criterion = nn.MSECriterion()

local trainer = NNTrainer(model, loopConfig, optimizer, optimConfig)
trainer:trainLoop(trainData, trainLabels, testData, testLabels)

----------------------------------------------------------------------
return lstm
