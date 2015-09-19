local rnn = {}

----------------------------------------------------------------------
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

----------------------------------------------------------------------
function rnn.backward(model, input, grad_out)
    local grad_state = torch.Tensor(grad_out[0]:size()):zero()
    local grad_input = {}
    for i = #model.replicas, 1, -1 do
        grad_input[i], grad_state = model.replicas[i]:backward(input[i], grad_out[i] + grad_state)
    end
    return grad_input, grad_state
end

----------------------------------------------------------------------
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
