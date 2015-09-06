local torch = require('torch')
local nn = require('nn')
local lstm = {}

----------------------------------------------------------------------
function lstm.createUnit(x, cPrev, hPrev)
    -- Calculate all four gates in one go
    local i2h = nn.Linear(params.rnn_size, 4 * params.rnn_size)(x)
    local h2h = nn.Linear(params.rnn_size, 4 * params.rnn_size)(hPrev)
    local gates = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshapedGates = nn.Reshape(4, params.rnn_size)(gates)
    local slicedGates = nn.SplitTable(2)(reshapedGates)

    -- Use select gate to fetch each gate and apply nonlinearity
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

----------------------------------------------------------------------
function lstm.createNetwork(inputs, numNode, timespan, inTimespan, outTimespan)
    local y = nn.Identity()()
    local statePrev = nn.Identity()()
    local stateNext = {}
    local split = {statePrev:split(2 * timespan)}
    for t = 1, inputTimespan do
        local cPrev = split[2 * t - 1]
        local hPrev = split[2 * t]
        local cellNext, hiddenNext = lstm.createUnit(x[t - 1], cPrev, hPrev)
        table.insert(stateNext, cellNext)
        table.insert(stateNext, hiddenNext)
        -- i[t] = hiddenNext
    end
    for t = inputTimespan, timespan do

    end
    local h2y = nn.Linear(params.rnn_size, params.vocab_size)
    -- local model = nn.gModule({x, y, statePrev},
    --                          {err, nn.Identity()(stateNext)})
    -- model:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

----------------------------------------------------------------------
return lstm
