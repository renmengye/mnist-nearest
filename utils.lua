local torch = require('torch')
local logger = require('logger')()
local progress = require('progress_bar')
local table_utils = require('table_utils')
local utils = {}

-------------------------------------------------------------------------------
function utils.getBatchIterator(data, labels, batchSize, printProgress)
    if printProgress == nil then
        printProgress = true
    end
    local step = 0
    local numData = data:size()[1]
    local numSteps = torch.ceil(numData / batchSize)
    local progressBar = progress.get(numSteps)
    return function()
               if step < numSteps then
                   local startIdx = batchSize * step + 1
                   local endIdx = batchSize * (step + 1)
                   if endIdx > numData then
                       endIdx = numData
                   end
                   local xBatch = data:index(
                       1, torch.range(startIdx, endIdx):long())
                   local labelBatch = labels:index(
                       1, torch.range(startIdx, endIdx):long())
                   step = step + 1
                   if printProgress then
                       progressBar(step)
                   end
                   return xBatch, labelBatch
               end
           end
end

-------------------------------------------------------------------------------
function utils.evalPrediction(prediction, labels)
    local correct = prediction:eq(labels):int():sum()
    local rate = correct / prediction:size()[1]
    labels = labels:reshape(labels:numel()):long()
    prediction = prediction:reshape(prediction:numel()):long()
    logger:logInfo(string.format('Accuracy: %.5f', rate))
    for i = 1,labels:size()[1] do
        logger:logInfo(
            string.format('Label: %d, Pred: %d', labels[i], prediction[i]), 1)
    end
    return rate
end

-------------------------------------------------------------------------------
function utils.getBag(N, max)
    local r = torch.rand(N) * max
    local items = torch.ceil(r):long()
    return items
end

-------------------------------------------------------------------------------
function utils.sliceLayer(parameterMap)
    logger:logInfo(table.tostring(parameterMap), 1)
    return function(vector, name)
        if parameterMap[name] == nil then
            logger:logFatal(string.format('key "%s" does not exist', name))
        end
        local i = parameterMap[name][1]
        local j = parameterMap[name][2]
        if i <= j then
            return vector[{{i, j}}]
        else
            return torch.Tensor()
        end
    end
end

-------------------------------------------------------------------------------
function utils.fillVector(vector, sliceLayer, valueTable)
    for key, value in pairs(valueTable) do
        sliceLayer(vector, key):fill(value)
        logger:logInfo(string.format('%s: %f', key, value))
    end
    return vector
end

-------------------------------------------------------------------------------
function utils.gradientClip(clipTable, sliceLayer)
    return function(dl_dw)
        function clip(x, cnorm)
            xnorm = torch.norm(x)
            logger:logInfo(string.format('Gradient norm: %.4f', xnorm), 2)
            if xnorm > cnorm then
                return x / xnorm * cnorm
            else
                return x
            end
        end
        dl_dw_clipped = torch.Tensor(dl_dw:size()):zero()
        if type(clipTable) == 'table' then
            for key, value in pairs(clipTable) do
                grad = sliceLayer(dl_dw, key)
                sliceLayer(dl_dw_clipped, key):copy(clip(grad, value))
            end
        else
            dl_dw_clipped = clip(dl_dw, clipTable)
        end
        return dl_dw_clipped
    end
end

-------------------------------------------------------------------------------
function utils.combineAllParameters(...)
    -- get parameters
    local networks = {...}
    if #networks == 1 and type(networks[1]) == 'table' then
        networks = networks[1]
    end
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()
        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = 
                storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = 
                        flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

-------------------------------------------------------------------------------
function utils.getParameterMap(params, names)
    local counter = 0
    local parameterMap = {}
    for k, net in ipairs(params) do
        local net_params, net_grads = net:parameters()
        parameterMap[names[k]] = {}
        parameterMap[names[k]][1] = counter + 1
        if net_params then
            for _, p in pairs(net_params) do
                logger:logInfo(string.format(
                    'node: %s start: %d #params: %d', 
                    names[k], counter, p:numel()))
                counter = counter + p:numel()
            end
        end
        parameterMap[names[k]][2] = counter
    end
    return parameterMap
end

-------------------------------------------------------------------------------
function utils.cloneModule(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

return utils
