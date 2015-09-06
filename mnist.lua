local torch = require('torch')
local Logger = require('logger')
local logger = Logger('mnist.lua', '')
local mnist = {}

----------------------------------------------------------------------
function pathJoin(path1, path2)
    if path1:sub(path1:len()) == '/' then
        return path1 .. path2
    else
        return path1 .. '/' .. path2
    end
end

----------------------------------------------------------------------
function mnist.loadData()
    logger:logInfo('Loading MNIST dataset')
    local folder = '../../data/mnist.t7'
    local trainFile = pathJoin(folder, 'train_32x32.t7')
    local testFile = pathJoin(folder, 'test_32x32.t7')
    local train = torch.load(trainFile, 'ascii')
    local test = torch.load(testFile, 'ascii')
    return train, test
end

----------------------------------------------------------------------
function mnist.floatNormalize(data, mean, std)
    local meanR, stdR
    if mean ~= nil then
        logger:logInfo('Mean is provided')
        meanR = mean:reshape(1, mean:numel())
    else
        meanR = nil
    end
    if std ~= nil then
        logger:logInfo('Std is provided')
        stdR = std:reshape(1, std:numel())
    else
        stdR = nil
    end
    dataR, meanR, stdR = mnist.flattenFloatNormalize(data, meanR, stdR)
    logger:logInfo('Unflatten data (N, 1024) -> (N, 1, 32, 32)')
    return dataR:reshape(data:size()), meanR:reshape(data:size()[3], data:size()[4]), stdR:reshape(data:size()[3], data:size()[4])
end

----------------------------------------------------------------------
function mnist.flattenFloat(data)
    logger:logInfo('Flatten data (N, 1, 32, 32) -> (N, 1024)')
    local dsize = data:size();
    return data:reshape(dsize[1], dsize[3] * dsize[4]):float()
end

----------------------------------------------------------------------
function mnist.flattenFloatNormalize(data, mean, std)
    dataR = mnist.flattenFloat(data)
    logger:logInfo('Normalize data: (data - mean) / std')
    if mean == nil then
        logger:logInfo('Calculate mean')
        mean = dataR:mean(1)
    end
    if std == nil then
        logger:logInfo('Calculate std')
        std = dataR:std(1)
        for i = 1,1024 do
            if std[1][i] == 0 then
                std[1][i] = 1.0
            end
        end
    end
    meanX = mean:expand(data:size()[1], 1024)
    stdX = std:expand(data:size()[1], 1024)
    return (dataR - meanX):cdiv(stdX), mean, std
end

----------------------------------------------------------------------
function mnist.visualize(data)
    logger:logInfo('Visualizing data')
    if itorch and data:numel() > 0 then
        if data:size()[1] > 256 then
            itorch.image(data[{{1,256}}])
        else
            itorch.image(data)
        end
    end
end

----------------------------------------------------------------------
return mnist
