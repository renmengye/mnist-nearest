local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
include 'embedding.lua'
local hdf5 = require('hdf5')
local nntrainer = require('nntrainer')
local Logger = require('logger')
local logger = Logger()

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

wordLength = 55
-- vocabSize = 97
-- numImages = 120
vocabSize = 9738
numImages = 123287
imageFeatLength = 4096
wordVecLength = 500

function createModel(imgFeat)
    local input = nn.Identity()()
    torch.setdefaulttensortype('torch.LongTensor')
    -- (B, 56) -> (B, 1)
    local imgSel = nn.Select(2, 1)(input)
    -- (B, 56) -> (B, 55)
    local txtSel = nn.Narrow(2, 2, wordLength)(input)
    torch.setdefaulttensortype('torch.FloatTensor')
    -- (B, 1) -> (B, 4096)
    local imgEmbedding = nn.Embedding(numImages, imageFeatLength, imgFeat)(imgSel)
    -- (B, 55) -> (B, 55, 500)
    local txtEmbedding = nn.Embedding(vocabSize, wordVecLength)(txtSel)
    -- (B, 55, 500) -> (B, 500)
    local bow = nn.Sum(2)(txtEmbedding)
    -- (B, 4096) + (B, 500) -> (B, 4596)
    local imgtxtConcat = nn.JoinTable(2, 2)({imgEmbedding, bow})
    -- (B, 4596) -> (B, 431)
    local answer = nn.Linear(imageFeatLength + wordVecLength, 431)(imgtxtConcat)
    return nn.gModule({input}, {answer})
end

logger:logInfo('Loading dataset')
local dataPath = '../../data/cocoqa-nearest/all_id.h5'
local data = hdf5.open(dataPath, 'r'):all()
local trainLabel = data.trainLabel + 1
local validLabel = data.validLabel + 1
local testLabel = data.testLabel + 1
local trainPlusValidData = torch.cat(data.trainData, data.validData, 1)
local trainPlusValidLabel = torch.cat(trainLabel, validLabel, 1)

logger:logInfo('Loading image features')
local imgFeatPath = '../../data/hidden_oxford_dense.h5'
local imgFeat = hdf5.open(imgFeatPath, 'r'):all()

logger:logInfo('Creating model')
local model = createModel(imgFeat.hidden7)

-- graph.dot(g.fg, 'Forward Graph', 'fg')
-- graph.dot(g.bg, 'Backward Graph', 'bg')

local loopConfig = {
    numEpoch = 15,
    trainBatchSize = 20,
    evalBatchSize = 1000
}
local optimConfig = {
    learningRate = 0.01,
    momentum = 0.9
}
local optimizer = optim.sgd

logger:logInfo('Start training')
nntrainer.trainAll(
    model, trainPlusValidData, trainPlusValidLabel, data.testData, testLabel, 
    loopConfig, optimizer, optimConfig)
