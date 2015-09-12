local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
include 'embedding.lua'
local hdf5 = require('hdf5')
local nntrainer = require('nntrainer')

function createModel()
    local model = nn.Sequential()
    model:add(nn.SparseLinear(10000, 2))
    model:add(nn.Linear(2, 3))
    return model
end

local model = createModel()
local x = torch.Tensor({ {1, 0.1}, {2, 0.3}, {10, 0.3}, {31, 0.2} })
local y = model:forward(x)
-- print(y)

-- 123287
-- 9738
batchSize = 1
wordLength = 3
-- vocabSize = 97
-- numImages = 120
vocabSize = 9738
numImages = 123287
imageFeatLength = 4096
wordVecLength = 500

imgFeatPath = '../../data/hidden_oxford_dense.h5'
imgFeat = hdf5.open(imgFeatPath, 'r'):all()

function createModel2()
    local input = nn.Identity()()
    local imgSel = nn.Select(2, 1)(input)
    local txtSel = nn.Narrow(2, 2, wordLength)(input)
    local txtSelReshape = nn.Reshape(batchSize * wordLength, false)(txtSel)
    local imgEmbedding = nn.Embedding(numImages, imageFeatLength, imgFeat.hidden7)(imgSel)
    local txtEmbedding = nn.Embedding(vocabSize, wordVecLength)(txtSelReshape)
    local txtEmbeddingReshape = nn.Reshape(batchSize, wordLength, wordVecLength)(txtEmbedding)
    local bow = nn.Sum(2)(txtEmbeddingReshape)
    local imgtxtConcat = nn.JoinTable(2, 2)({imgEmbedding, bow})
    local answer = nn.Linear(imageFeatLength + wordVecLength, 431)(imgtxtConcat)
    return nn.gModule({input}, {answer})
end

local g = createModel2()
graph.dot(g.fg, 'Forward Graph', 'fg')
graph.dot(g.bg, 'Backward Graph', 'bg')
a = torch.Tensor({{1, 2, 3, 4}})
-- print(g:forward(a))
----- Need to freeze layers

local dataPath = '../../data/cocoqa-nearest/all_id.h5'
local data = hdf5.open(dataIdPath, 'r'):all()
local trainLabel = data.trainLabel + 1
local validLabel = data.validLabel + 1
local testLabel = data.testLabel + 1
local trainPlusValidData = torch.cat(data.trainData, data.validData, 1)
local trainPlusValidLabel = torch.cat(trainLabel, validLabel, 1)
nntrainer.trainAll(
    model, trainPlusValidData, trainPlusValidLabel, data.testData, testLabel, 
    loopConfig, optimizer, optimConfig)
