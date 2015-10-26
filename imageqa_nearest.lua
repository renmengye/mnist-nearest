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
    local numTest = data.validData:size()[1]
    --local numTest = 200

    local processNearest
    if printNearestNeighbours then
        local dictPath
        if opt.dataset == 'cocoqa' then
            dictPath = '../image-qa/data/cocoqa/question_vocabs.txt'
        elseif opt.dataset == 'daquar' then
            dictPath = '../image-qa/data/daquar-37/question_vocabs.txt'
        end
        local qdict, iqdict = imageqa.readDict(dictPath)
        local dataId = imageqa.getid(opt.dataset)
        processNearest = function(id, neighbourIds)
            local example = imageqa.decodeSentences(
                dataId.validData[{id, {2, -1}}], iqdict)
            local neighbours = imageqa.decodeSentences(
                dataId.trainData:index(1, neighbourIds)[{{}, {2, -1}}], iqdict)
            logger:logInfo(string.format('T: %s', example))
            for i, s in ipairs(neighbours) do
                logger:logInfo(string.format('N: %s', s))
            end
        end
    else
        processNearest = nil
    end
    -- for k = 1, 61, 2 do
    --     local validPred = knn.runAll(
    --         k, data.trainData, data.trainLabel, data.validData, 
    --         numTest, printProgress, processNearest)
    --     local validLabelSubset = data.validLabel:index(1, 
    --         torch.range(1, numTest):long())
    --     local rate = utils.evalPrediction(validPred, validLabelSubset)
    --     if rate > bestRate then
    --         bestRate = rate
    --         bestK = k
    --     end
    -- end
    -- bestK = 13
    bestK = 31
    logger:logInfo(string.format('Best K is %d', bestK))

    logger:logInfo('Running on test set')
    --numTest = 10
    numTest = data.testData:size()[1]
    local testPred = knn.runAll(
        bestK, trainPlusValidData, trainPlusValidLabel, data.testData, numTest)
    local testLabelSubset = data.testLabel:index(1, torch.range(1, numTest):long())
    utils.evalPrediction(testPred, testLabelSubset)
    return testPred, testLabelSubset
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
cmd:option('-normimg', false, 'Whether to have the normalized image feature')
cmd:option('-normbow', false, 'Whether to have the normalized bow feature')
cmd:option('-trained_word_embed', false, 'Whether to use trained word embedding as BOW feature vector')

-- This should be better but may perform worse on smaller dataset like COCO-QA
cmd:option('-image_only', false, 'Only run on image features')
cmd:option('-text_only', false, 'Only run on BOW vectors')
cmd:option('-output', 'imageqa_nearest_out.txt', 'Output file')
cmd:option('-gt', 'imageqa_nearest_gt.txt', 'Ground truth file')
cmd:option('-dataset', 'cocoqa', 'Name of the dataset')
cmd:text()
opt = cmd:parse(arg)

logger:logInfo('--- command line options ---')
for key, value in pairs(opt) do
    logger:logInfo(string.format('%s: %s', key, value))
end
logger:logInfo('----------------------------')

local dataPath
local dataFolder = string.format('../../data/%s-nearest', opt.dataset)
if opt.normimg and opt.normbow then
    dataPath = string.format('%s/all_inorm_bnorm.h5', dataFolder)
elseif opt.normbow then
    dataPath = string.format('%s/all_iraw_bnorm.h5', dataFolder)
elseif opt.normimg then
    dataPath = string.format('%s/all_inorm_braw.h5', dataFolder)
else
    dataPath = string.format('%s/all_iraw_braw.h5', dataFolder)
end

logger:logInfo(string.format('Data: %s', dataPath))
local dataImgFeatureBowFeature = hdf5.open(dataPath, 'r'):all()

dataIdPath = string.format('%s/all_id.h5', dataFolder)
local dataImgIdBowId = hdf5.open(dataIdPath, 'r'):all()

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

collectgarbage()

local qdictPath, adictPath
if opt.dataset == 'cocoqa' then
    adictPath = '../image-qa/data/cocoqa/answer_vocabs.txt'
    qdictPath = '../image-qa/data/cocoqa/question_vocabs.txt'
elseif opt.dataset == 'daquar' then
    adictPath = '../image-qa/data/daquar-37/answer_vocabs.txt'
    qdictPath = '../image-qa/data/daquar-37/question_vocabs.txt'
else
    logger:logFatal(string.format('Unknown dataset: %s', opt.dataset))
end
local adict, iadict = imageqa.readDict(adictPath)
local qdict, iqdict = imageqa.readDict(qdictPath)

local testPred, testLabelSubset
testPred, testLabelSubset = run(data, true, false)
local outputFile = io.open(opt.output, 'w')
for i = 1, testPred:size(1) do
    outputFile:write(iadict[testPred[i] + 1])
    outputFile:write('\n')
end
outputFile:close()
local gtFile = io.open(opt.gt, 'w')
for i = 1, testLabelSubset:size(1) do
    local wordid = testLabelSubset[i][1] + 1
    local word = iadict[wordid]
    if word ~= nil then
        gtFile:write(word)
        gtFile:write('\n')
    else
        logger:logError(string.format('N: %d No found word: %d', i, wordid))
        gtFile:write('NILNIL\n')
    end
end
gtFile:close()

local qFile = io.open(string.format('%s_questions.txt', opt.dataset), 'w')
for i = 1, data.testLabel:size(1) do
    local example = imageqa.decodeSentences(
        dataImgIdBowId.testData[{i, {2, -1}}], iqdict)
    qFile:write(example)
    qFile:write('\n')
end
qFile:close()
