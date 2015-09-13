local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local hdf5 = require('hdf5')
local nntrainer = require('nntrainer')
local Logger = require('logger')
local logger = Logger()

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

wordLength = 55
vocabSize = 9739
numImages = 123287
imgFeatLength = 4096
wordVecLength = 500
numClasses = 431
wordEmbedInitRange = 1.0
linearInitRange = 0.01

function createModel()
    local input = nn.Identity()()
    -- (B, 4151) -> (B, 4096)
    local imgSel = nn.Narrow(2, 1, imgFeatLength)(input)
    -- (B, 56) -> (B, 55)
    local txtSel = nn.Narrow(2, imgFeatLength + 1, wordLength)(input)
    -- (B, 55) -> (B, 55, 500)
    local txtEmbeddingLayer = nn.LookupTable(vocabSize, wordVecLength)
    txtEmbeddingLayer.weight:copy(
        torch.rand(vocabSize, wordVecLength) * wordEmbedInitRange - wordEmbedInitRange / 2)
    local txtEmbeddingWeights = txtEmbeddingLayer.weight
    txtEmbedding = txtEmbeddingLayer(txtSel)
    -- (B, 55, 500) -> (B, 500)
    local bowLayer = nn.Sum(2)
    local bow = bowLayer(txtEmbedding)
    -- (B, 4096) + (B, 500) -> (B, 4596)
    local imgtxtConcat = nn.JoinTable(2, 2)({bow, imgSel})
    -- (B, 4596) -> (B, 431)
    local answerLayer = nn.Linear(imgFeatLength + wordVecLength, numClasses)
    local answerWeights =  answerLayer.weight
    local answerBias = answerLayer.bias
    answerLayer.weight:copy(
        torch.rand(imgFeatLength + wordVecLength, numClasses) * linearInitRange - linearInitRange / 2)
    answerLayer.bias:copy(torch.rand(numClasses) * linearInitRange - linearInitRange / 2)
    local answer = answerLayer(imgtxtConcat)
    return nn.gModule({input}, {answer}), txtEmbeddingWeights, answerWeights, answerBias, txtEmbeddingLayer, bowLayer, answerLayer
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('ImageQA IMG+BOW Training')
cmd:text()
cmd:text('Options:')
cmd:option('-train', false, 'whether to train a new network')
cmd:option('-path', 'imageqa_img_bow.w', 'save network path')
cmd:option('-save', false, 'whether to save the trained network')
cmd:option('-gpu', false, 'whether to run on GPU')
cmd:text()
opt = cmd:parse(arg)

if opt.gpu then
    require('cutorch')
    require('cunn')
end

logger:logInfo('Loading dataset')
local dataPath = '../../data/cocoqa-nearest/all_id_unk.h5'
local data = hdf5.open(dataPath, 'r'):all()

logger:logInfo('Loading image feature')
local dataImgPath = '../../data/cocoqa-nearest/img_feat.h5'
local dataImg = hdf5.open(dataImgPath, 'r'):all()

data.trainData = data.trainData[{{}, {2, 56}}]:float()
data.validData = data.validData[{{}, {2, 56}}]:float()
data.testData = data.testData[{{}, {2, 56}}]:float()

data.trainData = torch.cat(dataImg.train, data.trainData, 2)
data.validData = torch.cat(dataImg.valid, data.validData, 2)
data.testData = torch.cat(dataImg.test, data.testData, 2)

data.allData = torch.cat(data.trainData, data.validData, 1)
data.allLabel = torch.cat(data.trainLabel, data.validLabel, 1)

logger:logInfo('Creating model')
local model, txtEmbeddingWeights, answerWeights, answerBias, txtEmbeddingLayer, bowLayer, answerLayer = createModel()

-- graph.dot(g.fg, 'Forward Graph', 'fg')
-- graph.dot(g.bg, 'Backward Graph', 'bg')

local loopConfig = {
    numEpoch = 15,
    trainBatchSize = 20,
    evalBatchSize = 1000
}

local learningRates = torch.Tensor(model:getParameters():size()):zero()
learningRates:fill(1)
learningRates[{{1, txtEmbeddingWeights:numel()}}]:fill(80)

local optimConfig = {
    learningRate = 0.01,
    momentum = 0.9,
    learningRates = learningRates,
    gradientClip = 1.0
}

local optimizer = optim.sgd
logger:logInfo('Start training')
nntrainer.trainAll(
    model, data.allData, data.allLabel, data.testData, data.testLabel, 
    loopConfig, optimizer, optimConfig, opt.gpu)
print(nntrainer.evaluate(model, data.testData, data.testLabel, 100))
