local knn = require('nearest_neighbours')
local hdf5 = require('hdf5')
local utils = require('utils')
local Logger = require('logger')
local logger = Logger()
-- local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

function run(data)
    local trainPlusValidData = torch.cat(data.trainData, data.validData, 1)
    local trainPlusValidLabel = torch.cat(data.trainLabel, data.validLabel, 1)

    local bestK = 0
    local bestRate = -1.0
    logger:logInfo('Running on validation set')
    -- numTest = data.validData:size()[1]
    local numTest = 50
    for k = 21,61,2 do
        local validPred = knn.runAll(
            k, data.trainData, data.trainLabel, data.validData, numTest)
        local validLabelSubset = data.validLabel:index(1, torch.range(1, numTest):long())
        local rate = utils.evalPrediction(validPred, validLabelSubset)
        if rate > bestRate then
            bestRate = rate
            bestK = k
        end
    end
    logger:logInfo(string.format('Best K is %d', bestK))
    logger:logInfo(string.format('Best K is %d', bestK))

    logger:logInfo('Running on test set')
    numTest = data.testData:size()[1]
    local testPred = knn.runAll(
        bestK, trainPlusValidData, trainPlusValidLabel, data.testData, numTest)
    local testLabelSubset = data.testLabel:index(1, torch.range(1, numTest):long())
    utils.evalPrediction(testPred, testLabelSubset)
end

function getOneHot(data, vocabSize)
    logger:logInfo('Encoding one-hot vector')
    local dataFlatten = data:reshape(data:numel())
    local output = torch.Tensor(data:numel(), vocabSize):zero()
    for i = 1,data:numel() do
        output[i][dataFlatten[i]] = 1
    end

    local outputShape = torch.LongStorage(data:size():size() + 1)
    for i = 1,data:size():size() do
        outputShape[i] = data:size()[i]
    end
    outputShape[data:size():size() + 1] = vocabSize
    output = output:reshape(outputShape)
    return output
end

function getOneHotBOW(data, vocabSize)
    logger:logInfo('Encoding one-hot bow vector')
    local output = torch.Tensor(data:size()[1], vocabSize):zero()
    for i = 1,data:size()[1] do
        for j = 1,data:size()[2] do
            if data[i][j] > 0 then
                output[i][data[i][j]] = output[i][data[i][j]] + 1
            end
        end
    end
    return output
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('ImageQA Nearest Neighbours')
cmd:text()
cmd:text('Options:')
cmd:option('-normimg', false, 'whether to have the normalized image feature')
cmd:option('-normbow', false, 'whether to have the normalized bow feature')
cmd:option('-trained_word_embed', false, 'Whether to use trained word embedding as BOW feature vector')

-- This should be better but may perform worse on smaller dataset like COCO-QA
cmd:option('-image_only', false, 'Only run on image features')
cmd:option('-text_only', false, 'Only run on BOW vectors')
cmd:text()
opt = cmd:parse(arg)

local dataPath
if opt.normimg and opt.normbow then
    dataPath = '../../data/cocoqa-nearest/all_inorm_bnorm.h5'
elseif opt.normbow then
    dataPath = '../../data/cocoqa-nearest/all_iraw_bnorm.h5'
elseif opt.normimg then
    dataPath = '../../data/cocoqa-nearest/all_inorm_braw.h5'
else
    dataPath = '../../data/cocoqa-nearest/all_iraw_braw.h5'
end

logger:logInfo(string.format('Data: %s', dataPath))
local dataPath = '../../data/cocoqa-nearest/all_iraw_braw.h5'
local dataImgFeatureBowFeature = hdf5.open(dataPath, 'r'):all()

dataIdPath = '../../data/cocoqa-nearest/all_id.h5'
local dataImgIdBowId = hdf5.open(dataIdPath, 'r'):all()
for key, value in pairs(dataImgIdBowId) do
    print(key)
    print(value:size())
end

local x = torch.Tensor({{1,2,3,4}, {5,6,7,8}}):long()
print(getOneHot(x, 10))
print(getOneHotBOW(x, 10))

local data
if opt.trained_word_embed then
    -- Right now it is one-fold with image feature only
    if opt.image_only then
        logger:logInfo('Use image features only')
        data = {
            trainData = dataImgFeatureBowFeature.trainData:index(2, torch.range(1, 4096):long()),
            validData = dataImgFeatureBowFeature.validData:index(2, torch.range(1, 4096):long()),
            testData = dataImgFeatureBowFeature.testData:index(2, torch.range(1, 4096):long()),
            trainLabel = dataImgFeatureBowFeature.trainLabel,
            validLabel = dataImgFeatureBowFeature.validLabel,
            testLabel = dataImgFeatureBowFeature.testLabel
        }
    elseif opt.text_only then
        logger:logInfo('Use text features only')
        data = {
            trainData = dataImgFeatureBowFeature.trainData:index(2, torch.range(4097, 4596):long()),
            validData = dataImgFeatureBowFeature.validData:index(2, torch.range(4097, 4596):long()),
            testData = dataImgFeatureBowFeature.testData:index(2, torch.range(4097, 4596):long()),
            trainLabel = dataImgFeatureBowFeature.trainLabel,
            validLabel = dataImgFeatureBowFeature.validLabel,
            testLabel = dataImgFeatureBowFeature.testLabel
        }
    else
        logger:logInfo('Use trained word embedding')
        data = dataImgFeatureBowFeature
    end
else
    logger:logInfo('Use one-hot BOW vector')
    local numVocab = 9738
    local trainWordId = dataImgIdBowId.trainData:index(2, torch.range(2, dataImgIdBowId.trainData:size()[2]):long()):long()
    local trainBow = getOneHotBOW(trainWordId, 9738)

    local validWordId = dataImgIdBowId.validData:index(2, torch.range(2, dataImgIdBowId.validData:size()[2]):long()):long()
    local validBow = getOneHotBOW(validWordId, 9738)

    local testWordId = dataImgIdBowId.testData:index(2, torch.range(2, dataImgIdBowId.testData:size()[2]):long()):long()
    local testBow = getOneHotBOW(testWordId, 9738)
    if opt.image_only then
        logger:logInfo('Use image features only')
        data = {
            trainData = dataImgFeatureBowFeature.trainData:index(2, torch.range(1, 4096):long()),
            validData = dataImgFeatureBowFeature.validData:index(2, torch.range(1, 4096):long()),
            testData = dataImgFeatureBowFeature.testData:index(2, torch.range(1, 4096):long()),
            trainLabel = dataImgFeatureBowFeature.trainLabel,
            validLabel = dataImgFeatureBowFeature.validLabel,
            testLabel = dataImgFeatureBowFeature.testLabel
        }
    elseif opt.text_only then
        logger:logInfo('Use text features only')
        data = {
            trainData = trainBow,
            validData = validBow,
            testData = testBow,
            trainLabel = dataImgIdBowId.trainLabel,
            validLabel = dataImgIdBowId.validLabel,
            testLabel = dataImgIdBowId.testLabel
        }
    else
        data = {
            trainData = torch.cat(dataImgFeatureBowFeature.trainData:index(2, torch.range(1, 4096):long()),
                                  trainBow, 2),
            validData = torch.cat(dataImgFeatureBowFeature.validData:index(2, torch.range(1, 4096):long()),
                                  validBow, 2),
            testData = torch.cat(dataImgFeatureBowFeature.testData:index(2, torch.range(1, 4096):long()),
                                 testBow, 2),
            trainLabel = dataImgIdBowId.trainLabel,
            validLabel = dataImgIdBowId.validLabel,
            testLabel = dataImgIdBowId.testLabel
        }
        for key,value in pairs(data) do
            print(key)
            print(value:size())
        end
    end
end

collectgarbage()
run(data)
