local rnn = {}
local logger = require('logger')()
local table_utils = require('table_utils')
local torch = require('torch')

-------------------------------------------------------------------------------
local RNN, parent = torch.class('nn.RNN', 'nn.Module')

-------------------------------------------------------------------------------
function RNN:__init(coreUnit, timespan, interconnect)
    parent.__init(self)
    if interconnect == nil then
        interconnect = true
    end
    self.core = coreUnit
    self.timespan = timespan
    self.interconnect = interconnect
end

-------------------------------------------------------------------------------
function RNN:expand()
    self.replicas = RNN.cloneModule(self.core, self.timespan)
end

-------------------------------------------------------------------------------
function RNN:updateOutput(input)
    local output = {}
    local inputSeq, state0, global
    if self.name then
        -- print(self.name)
        -- print(input)
        -- print(#input)
        -- print(input[1])
        -- print(input[1][1])
        -- print(#input[1])
    end
    if self.interconnect then
        inputSeq, state0, global = unpack(input)
    else
        inputSeq = input
    end
    for t = 1, self.timespan do
        if t == 1 then
            if state0 == nil then
                output[t] = self.replicas[t]:forward(inputSeq[t])
            else
                output[t] = self.replicas[t]:forward({inputSeq[t], state0, global})
            end
        else
            if state0 == nil then
                output[t] = self.replicas[t]:forward(inputSeq[t])
            else
                output[t] = self.replicas[t]:forward({inputSeq[t], output[t - 1], global})
            end
        end
        -- print(output[t])
    end
    -- print('haha', output[1]:size())
    return output
end

-------------------------------------------------------------------------------
function RNN:updateGradInput(input, gradOutput)
    local gradInput = {}
    local gradState = torch.Tensor(gradOutput[1]:size()):zero()
    local inputSeq, state0, global
    if self.interconnect then
        inputSeq, state0, global = unpack(input)
    else
        inputSeq = input
    end
    local gradGlobal, gradGlobalTmp
    if global then
        gradGlobal = torch.Tensor(global:size()):zero()
    end
    for t = self.timespan, 1, -1 do
        local sum = gradOutput[t] + gradState
        if t == 1 then
            if state0 == nil then
                gradInput[t] = self.replicas[t]:updateGradInput(inputSeq[t], gradOutput[t])
            else
                gradInput[t], gradState, gradGlobalTmp = unpack(
                    self.replicas[t]:updateGradInput(
                        {inputSeq[t], state0, global}, sum))
            end
        else
            if state0 == nil then
                gradInput[t] = self.replicas[t]:updateGradInput(inputSeq[t], gradOutput[t])
            else
                gradInput[t], gradState, gradGlobalTmp = unpack(
                    self.replicas[t]:updateGradInput(
                        {inputSeq[t], self.replicas[t - 1].output, global}, sum))
            end
        end
        if global then
            gradGlobal:add(gradGlobalTmp)
        end
    end
    if global then
        self.gradInput = {gradInput, gradState, gradGlobal}
    else
        if state0 then
            self.gradInput = {gradInput, gradState}
        else
            self.gradInput = gradInput
        end
    end
    return self.gradInput
end

-------------------------------------------------------------------------------
function RNN:accGradParameters(input, gradOutput, scale)
    local gradInput = {}
    local gradState = torch.Tensor(gradOutput[1]:size()):zero()

    for t = self.timespan, 1, -1 do
        local gradSum = gradOutput[t] + gradState
        self.replicas[t]:accGradParameters(input[t], gradSum)
        if self.interconnect then
            gradState = self.replicas[t].gradInput[2]
        end
    end
end

-------------------------------------------------------------------------------
function RNN:parameters()
    return self.core:parameters()
end

-------------------------------------------------------------------------------
function RNN.cloneModule(net, T)
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

function RNN:train()
    for i, r in ipairs(self.replicas) do
        r:train()
    end
end

function RNN:evaluate()
    for i, r in ipairs(self.replicas) do
        r:evaluate()
    end
end

-------------------------------------------------------------------------------
function rnn.forward(model, input, state0)
    local output = {}
    for i = 1, #model.replicas do
        if i == 1 then
            output[i] = model.replicas[i]:forward({input[i], state0})
        else
            output[i] = model.replicas[i]:forward({input[i], output[i - 1]})
        end
    end
    return output
end

-------------------------------------------------------------------------------
function rnn.backward(model, input, grad_out)
    local grad_input = {}
    local grad_state = torch.Tensor(grad_out[1]:size()):zero()

    for i = #model.replicas, 1, -1 do
        local sum = grad_out[i] + grad_state
        grad_input[i], grad_state = unpack(
            model.replicas[i]:backward(input[i], sum))
    end
    return grad_input, grad_state
end

-------------------------------------------------------------------------------
function rnn.cloneModule(net, T)
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

return rnn
