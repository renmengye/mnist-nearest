local torch = require('torch')
local nn = require('nn')
local lstm = {}

----------------------------------------------------------------------
function lstm.createUnit(x, cPrev, hPrev)
    -- Share weights???
    local i2h = nn.Linear(params.rnn_size, 4 * params.rnn_size)(x)
    local h2h = nn.Linear(params.rnn_size, 4 * params.rnn_size)(hPrev)
    local gates = nn.CAddTable()({i2h, h2h})
    local reshapedGates = nn.Reshape(4, params.rnn_size)(gates)
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

----------------------------------------------------------------------
function lstm.createNetwork(inputs, numNode, timespan, inEnd, outStart)
    -- At training time, inEnd should usually be equal to timespan.
    -- You will need to do masking afterwards, if you want variable length 
    -- output in a mini-batch.
    local statePrev = nn.Identity()() -- no input??
    local stateNext = {}
    local split = {statePrev:split(2 * timespan)}
    for t = 1,inEnd do
        local cPrev = split[2 * t - 1]
        local hPrev = split[2 * t]
        local cellNext, hiddenNext = lstm.createUnit(x[t - 1], cPrev, hPrev)
        table.insert(stateNext, cellNext)
        table.insert(stateNext, hiddenNext)
        -- i[t] = hiddenNext
    end
    if inEnd < outStart then
        -- Blank region between input ends and output begins
        -- This is pretty useless...
        for t = inEnd,outStart do
        end
        -- Pure output, feed output back in.
        for t = outStart,timespan do
            local cPrev = split[2 * t - 1]
            local hPrev = split[2 * t]
            local cellNext, hiddenNext = lstm.createUnit(x[t - 1], cPrev, hPrev)
            table.insert(stateNext, cellNext)
            table.insert(stateNext, hiddenNext)
        end
    else
        -- Pure output, feed output back in.
        for t = inEnd,timespan do
        end
    end
end

-- write a good model weights serializer: a dictionary!!
-- flatten weights doesn't really make sense other than optimization
-- for example conv net... flatten weights is really architecture dependent.

----------------------------------------------------------------------
return lstm
