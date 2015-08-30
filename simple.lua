require('torch')
require('io')

trainFile = '../../data/mnist.t7/train_32x32.t7'
testFile = '../../data/mnist.t7/test_32x32.t7'

print('==> loading dataset')

-- We load the dataset from disk, it's straightforward
train = torch.load(trainFile, 'ascii')
test = torch.load(testFile, 'ascii')

-- (N, 1, 32, 32) -> (N, 1024)
trainSize = train.data:size()
trainDataResize = train.data:reshape(trainSize[1], trainSize[3] * trainSize[4])
testSize = test.data:size()
testDataResize = test.data:reshape(testSize[1], testSize[3] * testSize[4])

----------------------------------------------------------------------
print('==> visualizing data')

-- Visualization is quite easy, using itorch.image().
if itorch then
   print('training data:')
   itorch.image(train.data[{ {1,256} }])
   print('test data:')
   itorch.image(test.data[{ {1,256} }])
end

function distance(a, b)
    d = (a - b):float()
    return d:cmul(d):sum()
end

function distanceM(A, b)
    local B, D
    b = b:float()
    A = A:float()
    B = b:reshape(1, b:size()[1])
    B = B:expand(A:size()[1], b:size()[1])
    D = (A - B)
    D = D:cmul(D):sum(2)
    return D
end

function nearestNeighbour(data, labels, example, k)
    local N = data:size()[1]
    local minDist = distance(data[1], example)
    local minLabel = labels[1]
    local dist = 0
    distAll = distanceM(data, example)
    minDist, minId = distAll:min(1)
    return labels[minId[1][1]]
end

numTest = 10
print(string.format('==> running %d test examples', numTest))
prediction = torch.ByteTensor(numTest)
progress = 0
for i = 1,numTest do
    prediction[i] = nearestNeighbour(trainDataResize, train.labels, testDataResize[i], 1)
    -- print(string.format('Example: %d, Pred: %d, GT: %d', i, prediction[i], test.labels[i]))
    collectgarbage()
    while i / numTest > progress / 80 do
        io.write('.')
        io.flush()
        progress = progress + 1
    end
end

testSub = test.labels:index(1, torch.range(1, numTest):long())
correct = prediction:eq(testSub):int():sum()
print(string.format('accuracy: %.5f', correct / numTest))
