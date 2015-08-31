require('torch')

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
    print('==> loading MNIST dataset')
    local folder = '../../data/mnist.t7'
    local trainFile = pathJoin(folder, 'train_32x32.t7')
    local testFile = pathJoin(folder, 'test_32x32.t7')
    local train = torch.load(trainFile, 'ascii')
    local test = torch.load(testFile, 'ascii')
    return train, test
end

----------------------------------------------------------------------
function mnist.flattenFloat(data)
    print('==> flatten data (N, 1, 32, 32) -> (N, 1024)')
    local dsize = data:size();
    return data:reshape(dsize[1], dsize[3] * dsize[4]):float()
end

----------------------------------------------------------------------
function mnist.visualize(data)
    print('==> visualizing data')
    if itorch and data:numel() > 0 then
        if data:size()[1] > 256 then
            itorch.image(data[{{1,256}}])
        else
            itorch.image(data)
        end
    end
end

return mnist
