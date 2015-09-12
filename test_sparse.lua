local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
-- include 'embedding.lua'
local hdf5 = require('hdf5')
local nntrainer = require('nntrainer')
local Logger = require('logger')
local logger = Logger()

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

wordLength = 55
vocabSize = 9739
numImages = 123287
imgFeatLength = 4096
wordVecLength = 500

function createModel()
    local input = nn.Identity()()
    -- (B, 4151) -> (B, 4096)
    local imgSel = nn.Narrow(2, 1, imgFeatLength)(input)
    -- (B, 56) -> (B, 55)
    local txtSel = nn.Narrow(2, 4097, wordLength)(input)
    -- (B, 55) -> (B, 55, 500)
    local txtEmbedding = nn.LookupTable(vocabSize, wordVecLength)(txtSel)
    -- (B, 55, 500) -> (B, 500)
    local bow = nn.Sum(2)(txtEmbedding)
    -- (B, 4096) + (B, 500) -> (B, 4596)
    local imgtxtConcat = nn.JoinTable(2, 2)({imgSel, bow})
    -- (B, 4596) -> (B, 431)
    local answer = nn.Linear(imgFeatLength + wordVecLength, 431)(imgtxtConcat)
    return nn.gModule({input}, {answer})
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

data.trainData = data.trainData[{{}, {2, 56}}]
data.validData = data.validData[{{}, {2, 56}}]
data.testData = data.testData[{{}, {2, 56}}]

data.trainData = data.trainData:float()
data.validData = data.validData:float()
data.testData = data.testData:float()

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
    model, data.allData, data.allLabel, data.testData, data.testLabel, 
    loopConfig, optimizer, optimConfig, opt.gpu)
