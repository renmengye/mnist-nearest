local knn = require('nearest_neighbours')
local hdf5 = require('hdf5')
local utils = require('utils')
local Logger = require('logger')
local logger = Logger()
-- local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('ImageQA Nearest Neighbours')
cmd:text()
cmd:text('Options:')
cmd:option('-normimg', false, 'whether to have the normalized image feature')
cmd:option('-normbow', false, 'whether to have the normalized bow feature')
cmd:text()
opt = cmd:parse(arg)

local dataPath
if opt.normimg and opt.normbow then
-- local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'
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
local data = hdf5.open(dataPath, 'r'):all()
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
