local hdf5 = require('hdf5')
local lunajson = require('lunajson')
local torch = require('torch')
local logger = require('logger')()
local table_utils = require('table_utils')
local imageqa = {}

----------------------------------------------------------------------

function imageqa.readImgDictCocoqa(imgidDict)
    local jsonTrainFilename = '/ais/gobi3/datasets/mscoco/annotations/captions_train2014.json'
    local jsonValidFilename = '/ais/gobi3/datasets/mscoco/annotations/captions_val2014.json'
    local urlDict = {}
    local fh = io.open(jsonTrainFilename)
    local captiontxt = fh:read('*a')
    local caption = lunajson.decode(captiontxt)
    fh:close()
    for i, item in ipairs(caption.images) do
        urlDict[item.id] = item.url
    end
    fh = io.open(jsonValidFilename)
    captiontxt = fh:read('*a')
    caption = lunajson.decode(captiontxt)
    fh:close()
    for i, item in ipairs(caption.images) do
        urlDict[item.id] = item.url
    end
    local urlList = {}
    for i, key in ipairs(imgidDict) do
        urlList[i] = urlDict[tonumber(key)]
    end
    return urlList
end

----------------------------------------------------------------------

function imageqa.readDict(filename)
    local fh, err = io.open(filename)
    local dict = {}
    local idict = {}
    if err then
        logger:logError(string.format('Error reading %s', filename))
        return
    end
    local index = 1
    while true do
        line = fh:read()
        if line == nil then
            break
        end
        dict[line] = index
        table.insert(idict, line)
        logger:logInfo(string.format('%s: %d', line, index), 2)
        index = index + 1
    end
    fh:close()
    return dict, idict
end

----------------------------------------------------------------------

function imageqa.saveDict(idict, filename)
    local fh = io.open(filename, 'w')
    for i, word in ipairs(idict) do
        fh:write(string.format('%s\n', word))
    end
    fh:close()
end

----------------------------------------------------------------------

function imageqa.invertDict(dict)
    local idict = {}
    for k, v in pairs(dict) do
        idict[v] = k
    end
    return idict
end

----------------------------------------------------------------------

function imageqa.decodeSentences(ids, idict)
    local id_size = ids:size()
    local result = {}
    local resultstr
    if id_size:size() == 1 then
        for i = 1, id_size[1] do
            local id = ids[i]
            if id > 0 then
                table.insert(result, idict[id])
            else
                break
            end
        end
        resultstr = table.concat(result, ' ')
    elseif id_size:size() == 2 then
        resultstr = {}
        for i = 1, id_size[1] do
            for j = 1, id_size[2] do
                local id = ids[i][j]
                if id > 0 then
                    table.insert(result, idict[id])
                else
                    break
                end
            end
            table.insert(resultstr, table.concat(result, ' '))
            result = {}
            collectgarbage()
        end
    end
    return resultstr
end

----------------------------------------------------------------------

function imageqa.encodeSentences(sentences, dict, chunk)
    if chunk == nil then
        chunk = false
    end
    local function findMaxLen(sentences)
        if type(sentences) == 'string' then
            local count = 0
            for word in string.gmatch(sentences, '([^%s]+)') do
                count = count + 1
            end
            return count
        elseif type(sentences) == 'table' then
            local maxlen = 0
            for i, v in ipairs(sentences) do
                local len = findMaxLen(v)
                if len > maxlen then
                    maxlen = len
                end
            end
            return maxlen
        end
    end

    local function encodeSingle(sentence, dict, maxlen)
        if maxlen == nil then
            maxlen = findMaxLen(sentence)
        end
        local result = torch.Tensor(maxlen):zero()
        local index = 1
        for word in string.gmatch(sentence, '([^%s]+)') do
            result[index] = dict[word]
            index = index + 1
        end
        return result
    end

    if type(sentences) == 'string' then
        return encodeSingle(sentences, dict)
    elseif type(sentences) == 'table' then
        local maxlen
        local result
        if chunk then
            maxlen = findMaxLen(sentences)
            result = torch.Tensor(#sentences, maxlen):zero()
        else
            maxlen = nil
            result = {}
        end

        for i, v in ipairs(sentences) do
            result[i] = encodeSingle(v, dict, maxlen)
        end
        collectgarbage()
        return result
    end
end

----------------------------------------------------------------------

function imageqa.getid(dataset)
    if dataset == 'cocoqa' then
        dataPath = '../../data/cocoqa-nearest/all_id.h5'
    elseif dataset == 'daquar-37' then
        dataPath = ''
    else
        logger:logFatal(string.format('No dataset found: %s', dataset))
    end
    local data = hdf5.open(dataPath, 'r'):all()
    return data
end

----------------------------------------------------------------------

-- local qdict, iqdict = imageqa.readDict('../image-qa/data/cocoqa/question_vocabs.txt')
-- local data = imageqa.getid('cocoqa')
-- logger:logInfo(imageqa.decodeSentences(data.trainData[{1, {2, 56}}], iqdict))
-- local sentences = {}
-- for i, sentence in ipairs(imageqa.decodeSentences(data.trainData[{{1, 5}, {2, 56}}], iqdict)) do
--     table.insert(sentences, sentence)
-- end
-- logger:logInfo(table.tostring(imageqa.encodeSentences(sentences, qdict)))
-- logger:logInfo(imageqa.encodeSentences(sentences, qdict, true))

return imageqa
