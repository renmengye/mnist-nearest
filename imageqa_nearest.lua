local imageqa = require('imageqa')
local knn = require('nearest_neighbours')
local hdf5 = require('hdf5')
local utils = require('utils')
local Logger = require('logger')
local logger = Logger()

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

function run(data, printProgress, printNearestNeighbours)
    if printNearestNeighbours == nil then
        printNearestNeighbours = false
    end
    local trainPlusValidData = torch.cat(data.trainData, data.validData, 1)
    local trainPlusValidLabel = torch.cat(data.trainLabel, data.validLabel, 1)

    local bestK = 0
    local bestRate = -1.0
    logger:logInfo('Running on validation set')
    -- numTest = data.validData:size()[1]
    local numTest = 200

    local processNearest
    if printNearestNeighbours then
        local dictPath = '../image-qa/data/cocoqa/question_vocabs.txt'
        local qdict, iqdict = imageqa.readDict(dictPath)
        local dataId = imageqa.getid('cocoqa')
        processNearest = function(id, neighbourIds)
            local example = imageqa.decodeSentence(dataId. validData[{id, {2, 56}}], iqdict)
            local neighbours = imageqa.decodeSentence(dataId.trainData:index(1, neighbourIds)[{{}, {2, 56}}], iqdict)
            logger:logInfo(string.format('T: %s', example))
            for i, s in ipairs(neighbours) do
                logger:logInfo(string.format('N: %s', s))
            end
        end
    else
        processNearest = nil
    end
    for k = 1, 61, 2 do
        local validPred = knn.runAll(
            k, data.trainData, data.trainLabel, data.validData, numTest, printProgress, processNearest)
        local validLabelSubset = data.validLabel:index(1, torch.range(1, numTest):long())
        local rate = utils.evalPrediction(validPred, validLabelSubset)
        if rate > bestRate then
            bestRate = rate
            bestK = k
        end
    end
    logger:logInfo(string.format('Best K is %d', bestK))

    logger:logInfo('Running on test set')
    numTest = data.testData:size()[1]
    local testPred = knn.runAll(
        bestK, trainPlusValidData, trainPlusValidLabel, data.testData, numTest)
    local testLabelSubset = data.testLabel:index(1, torch.range(1, numTest):long())
    utils.evalPrediction(testPred, testLabelSubset)
end

function getOneHot(data, vocabSize)
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
local dataImgFeatureBowFeature = hdf5.open(dataPath, 'r'):all()

dataIdPath = '../../data/cocoqa-nearest/all_id.h5'
local dataImgIdBowId = hdf5.open(dataIdPath, 'r'):all()
local data

if opt.image_only then
    logger:logInfo('Use image features only')
    data = {
        trainData = dataImgFeatureBowFeature.trainData[{{}, {1, 4096}}],
        validData = dataImgFeatureBowFeature.validData[{{}, {1, 4096}}],
        testData = dataImgFeatureBowFeature.testData[{{}, {1, 4096}}],
        trainLabel = dataImgFeatureBowFeature.trainLabel,
        validLabel = dataImgFeatureBowFeature.validLabel,
        testLabel = dataImgFeatureBowFeature.testLabel
    }
elseif opt.trained_word_embed then
    logger:logInfo('Use trained word embedding')
    if opt.text_only then
        logger:logInfo('Use text features only')
        data = {
            trainData = dataImgFeatureBowFeature.trainData[{{}, {4097, 4596}}],
            validData = dataImgFeatureBowFeature.validData[{{}, {4097, 4596}}],
            testData = dataImgFeatureBowFeature.testData[{{}, {4097, 4596}}],
            trainLabel = dataImgFeatureBowFeature.trainLabel,
            validLabel = dataImgFeatureBowFeature.validLabel,
            testLabel = dataImgFeatureBowFeature.testLabel
        }
    else
        data = dataImgFeatureBowFeature
    end
else
    logger:logInfo('Encoding one-hot vector')
    local numVocab = 9738
    local trainWordId = dataImgIdBowId.trainData[{{}, {2, dataImgIdBowId.trainData:size(2)}}]
    local trainBow = getOneHotBOW(trainWordId, 9738)
    local validWordId = dataImgIdBowId.validData[{{}, {2, dataImgIdBowId.validData:size(2)}}]
    local validBow = getOneHotBOW(validWordId, 9738)
    local testWordId = dataImgIdBowId.testData[{{}, {2, dataImgIdBowId.testData:size(2)}}]
    local testBow = getOneHotBOW(testWordId, 9738)
    if opt.text_only then
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
            trainData = torch.cat(
                dataImgFeatureBowFeature.trainData[{{}, {1, 4096}}],
                trainBow, 2),
            validData = torch.cat(
                dataImgFeatureBowFeature.validData[{{}, {1, 4096}}],
                validBow, 2),
            testData = torch.cat(
                dataImgFeatureBowFeature.testData[{{}, {1, 4096}}],
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

-- print(data.trainData[1])
collectgarbage()
run(data, false, true)
