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

-- Hyper-parameters
wordLength = 55
vocabSize = 9739
numImages = 123287
imgFeatLength = 4096
wordVecLength = 500
numClasses = 431
wordEmbedInitRange = 1.0
wordEmbedLearningRate = 0.8
wordEmbedGradientClip = 0.1
answerInitRange = 0.01
answerLearningRate = 0.1
answerGradientClip = 0.1
answerWeightDecay = 0.00005
momentum = 0.9
dropoutRate = 0.5

-- Model architecture
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
    txtEmbedding = txtEmbeddingLayer(txtSel)
    -- (B, 55, 500) -> (B, 500)
    local bowLayer = nn.Sum(2)
    local bow = bowLayer(txtEmbedding)
    -- (B, 4096) + (B, 500) -> (B, 4596)
    local imgtxtConcat = nn.JoinTable(2, 2)({bow, imgSel})

    local dropout
    if dropoutRate > 0.0 then
        local dropoutLayer = nn.Dropout(dropoutRate)
        dropout = dropoutLayer(imgtxtConcat)
    else
        dropout = imgtxtConcat
    end
    -- (B, 4596) -> (B, 431)
    local answerLayer = nn.Linear(imgFeatLength + wordVecLength, numClasses)
    answerLayer.weight:copy(
        torch.rand(imgFeatLength + wordVecLength, numClasses) * answerInitRange - answerInitRange / 2)
    answerLayer.bias:copy(torch.rand(numClasses) * answerInitRange - answerInitRange / 2)
    local answer = answerLayer(dropout)
    return nn.gModule({input}, {answer})
end

-- Command line options
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

-- Load data
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
local model = createModel()

-- graph.dot(g.fg, 'Forward Graph', 'fg')
-- graph.dot(g.bg, 'Backward Graph', 'bg')

local loopConfig = {
    numEpoch = 200,
    trainBatchSize = 64,
    evalBatchSize = 1000
}

-- Helper function to slice the entire vector
local function sliceLayer(vector, name)
    if name == 'txtEmbedding' then
        return vector[{{1, vocabSize * wordVecLength}}]
    elseif name == 'answer' then
        return vector[{{vocabSize * wordVecLength + 1, vector:numel()}}]
    else
        logger:logError(string.format('No layer with name %s found', name))
    end
end

-- Set up different learning rates for each layer
local learningRates = torch.Tensor(model:getParameters():size()):zero()
learningRates:fill(answerLearningRate)
sliceLayer(learningRates, 'txtEmbedding'):fill(wordEmbedLearningRate)

-- Set up different weight decays for each layer
local weightDecays = torch.Tensor(model:getParameters():size()):zero()
weightDecays:fill(0.0)
sliceLayer(weightDecays, 'answer'):fill(answerWeightDecay)

-- Set up different gradient clipping for each layer
local gradientClip = function(dl_dw)
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
    txtEmbeddingGrad = sliceLayer(dl_dw, 'txtEmbedding')
    sliceLayer(dl_dw_clipped, 'txtEmbedding'):copy(
        clip(txtEmbeddingGrad, wordEmbedGradientClip))
    answerGrad = sliceLayer(dl_dw, 'answer')
    sliceLayer(dl_dw_clipped, 'answer'):copy(
        clip(answerGrad, answerGradientClip))
    return dl_dw_clipped
end

-- Construct optimizer configs
local optimConfig = {
    learningRate = 1.0,
    momentum = momentum,
    learningRates = learningRates,
    weightDecay = 0.0,
    weightDecays = weightDecays,
    gradientClip = gradientClip
}

local optimizer = optim.sgd
logger:logInfo('Start training')
nntrainer.trainAll(
    model, data.allData, data.allLabel, data.testData, data.testLabel, 
    loopConfig, optimizer, optimConfig, opt.gpu)
print(nntrainer.evaluate(model, data.testData, data.testLabel, 100))
