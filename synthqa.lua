local torch = require('torch')
local logger = require('logger')()
local table_utils = require('table_utils')
local imageqa = require('imageqa')
local utils = require('utils')
local nnevaluator = require('nnevaluator')
local synthqa = {}

-------------------------------------------------------------------------------
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

-------------------------------------------------------------------------------
synthqa.OBJECT = {
    'circle',
    'square',
    'triangle'
}

synthqa.X = {
    'top', 
    'middle', 
    'bottom'
}

synthqa.Y = {
    'left', 
    'center', 
    'right'
}

synthqa.COLOR = {
    'red', 
    'orange', 
    'yellow', 
    'green', 
    'blue', 
    'purple', 
    'black'
}

synthqa.RELATION = {
    'above', 
    'below', 
    'left_of', 
    'right_of'
}

synthqa.WORDS = {
    'what',
    'how',
    'many',
    'color',
    'is'
}

synthqa.NUMBER = {
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
}

synthqa.NUM_GRID = #synthqa.X * #synthqa.Y
synthqa.OBJECT_SIZE_AVG = 0.1
synthqa.OBJECT_SIZE_STD = 0.03
synthqa.X_STD = 0.05
synthqa.Y_STD = 0.05
synthqa.HAS_DUPLICATES = true
synthqa.DUPLICATES_LAMBDA = 1.0

-------------------------------------------------------------------------------
function combine(tables)
    local combined = {}
    local count = 1
    for i, t in ipairs(tables) do
        for j, v in ipairs(t) do
            combined[v] = count
            count = count + 1
        end
    end
    return combined
end

-------------------------------------------------------------------------------
function synthqa.genHowManyObject(N)
    local dataset = {}
    local maxNumItems = 0
    for i = 1, N do
        local objectOfInterest = i % #synthqa.OBJECT + 1
        local question = string.format(
            'how many %s', synthqa.OBJECT[objectOfInterest])
        local items = {}
        local numObj = {}

        -- Empty object is encoded as the largest object ID.
        for j = 1, #synthqa.OBJECT + 1 do
            numObj[j] = 0
        end
        for j = 1, synthqa.NUM_GRID do
            local objCat = torch.ceil(torch.uniform() * (#synthqa.OBJECT + 1))
            numObj[objCat] = numObj[objCat] + 1
            table.insert(items, {category = objCat, color = 1, grid = j})
            -- Each items have some dupelicates.
            -- Number of duplicates
            if synthqa.HAS_DUPLICATES then
                local numDup = torch.exponential(synthqa.DUPLICATES_LAMBDA)
                if numDup > 0 then
                    for k = 1, numDup do
                        table.insert(items, 
                            {category = objCat, color = 1, grid = j})
                    end
                end
            end
        end
        local answer = synthqa.NUMBER[numObj[objectOfInterest] + 1]
        if #items > maxNumItems then maxNumItems = #items end
        logger:logInfo(
            string.format(
                'N1: %d, N2: %d, N3: %d, N4:%d, N: %d, NN: %d', 
                numObj[1], numObj[2], numObj[3], numObj[4],
                numObj[1] + numObj[2] + numObj[3],
                #items), 2)
        logger:logInfo(
            string.format(
                'Q: %s, A: %s', 
                question, answer), 2)
        logger:logInfo(table.tostring(items), 2)
        table.insert(dataset, {
            question = question,
            answer = answer,
            items = items
        })
    end
    dataset.maxNumItems = maxNumItems
    dataset.hasDuplicate = synthqa.HAS_DUPLICATES
    logger:logInfo(string.format('Max number of items: %d', maxNumItems), 2)
    return dataset
end

-------------------------------------------------------------------------------
function synthqa.getCoord(grid, noise)
    if noise == nil then
        noise = true
    end
    -- Randomly sample with gaussian noise
    local yCenter = torch.floor((grid - 1) / 3)  / 3 + (1 / 6)
    local xCenter = (grid - 1) % 3 / 3 + (1 / 6)
    if noise then
        logger:logInfo(
            string.format(
                'Grid %d, X center %f Y center %f', 
                grid, xCenter, yCenter), 2)
        return torch.Tensor({torch.normal(xCenter, synthqa.X_STD),
                torch.normal(yCenter, synthqa.Y_STD),
                torch.normal(
                    synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD),
                torch.normal(
                    synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD)})
    else
        return torch.Tensor(
            {xCenter, yCenter, 
            synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_AVG})
    end
end

-------------------------------------------------------------------------------
function synthqa.encodeItems(datasetItems, numItems, encodeGridId)
    -- Category ID (1)
    -- Color ID (1)
    -- X, Y coordinates (2)
    -- Unique grid ID (for duplicates removal groundtruth)
    -- local colorIdict = imageqa.invertDict(COLOR)
    -- local objTypeIdict = imageqa.invertDict(OBJECT)
    if encodeGridId == nil then
        encodeGridId = true
    end
    local function getCoord(grid, noise)
        if noise == nil then
            noise = true
        end
        -- Randomly sample with gaussian noise
        local yCenter = torch.floor((grid - 1) / 3)  / 3 + (1 / 6)
        local xCenter = (grid - 1) % 3 / 3 + (1 / 6)
        if noise then
            logger:logInfo(
                string.format(
                    'Grid %d, X center %f Y center %f', 
                    grid, xCenter, yCenter), 2)
            return torch.Tensor({torch.normal(xCenter, synthqa.X_STD),
                    torch.normal(yCenter, synthqa.Y_STD),
                    torch.normal(
                        synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD),
                    torch.normal(
                        synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD)})
        else
            return torch.Tensor(
                {xCenter, yCenter, 
                synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_AVG})
        end
    end
    
    local numDim
    if encodeGridId then
        numDim = 7
    else
        numDim = 6
    end
    local result = torch.Tensor(
        #datasetItems, numItems * numDim):zero()
    for i, items in ipairs(datasetItems) do
        if #items > 0 then
            local itemShuffle = torch.randperm(#items)
            -- Shuffle items
            for j = 1, #items do
                local item = items[itemShuffle[j]]
                local itemCount = 0
                for key, value in pairs(item) do
                    if key == 'category' then
                        result[{i, (j - 1) * numDim + 1}] = value
                    elseif key == 'color' then
                        result[{i, (j - 1) * numDim + 2}] = value
                    elseif key == 'grid' then
                        result[
                        {i, {(j - 1) * numDim + 3, (j - 1) * numDim + 6}}] = 
                        getCoord(value)
                        if encodeGridId then
                            result[{i, (j - 1) * numDim + 7}] = value
                        end
                    end
                end
            end
            if #items < numItems then
                for j = #items + 1, numItems do
                    result[{i, {(j - 1) * numDim + 1, j * numDim}}] = 
                        torch.Tensor({4, 1, 1, 1, 1, 1, 10})
                end
            end
        end
    end
    return result
end

-------------------------------------------------------------------------------
synthqa.dict = combine({
            synthqa.OBJECT, 
            synthqa.X, 
            synthqa.Y, 
            synthqa.COLOR, 
            synthqa.RELATION, 
            synthqa.WORDS,
            synthqa.NUMBER
        })
synthqa.idict = imageqa.invertDict(synthqa.dict)

-------------------------------------------------------------------------------
function synthqa.prep(dataset, objective)
    if objective == nil then
        objective = 'classification'
    end
    logger:logInfo(string.format('dataset label encoding: %s', objective))
    logger:logInfo(table.tostring(synthqa.dict), 2)
    logger:logInfo(table.tostring(synthqa.idict), 2)
    local questions = {}
    local answers = {}
    local datasetItems = {}
    for i, entry in ipairs(dataset) do
        table.insert(questions, entry.question)
        table.insert(answers, entry.answer)
        table.insert(datasetItems, entry.items)
    end

    -- N x Q
    local questionIds = imageqa.encodeSentences(questions, synthqa.dict, true)
    logger:logInfo(string.format('question length: %d', questionIds:size(2)))
    -- N
    local answerIds = 
        imageqa.encodeSentences(answers, synthqa.dict, true):reshape(#answers)
    -- N x 54 or N x 63 (encode grid id)
    local itemIds = synthqa.encodeItems(
        datasetItems, dataset.maxNumItems, true)
    logger:logInfo('encoded questions', 2)
    logger:logInfo(questionIds, 2)
    logger:logInfo('encoded answers', 2)
    logger:logInfo(answerIds, 2)
    logger:logInfo('encoded items', 2)
    logger:logInfo(itemIds:reshape(
        #answers, dataset.maxNumItems, 
        itemIds:size(2) / dataset.maxNumItems), 2)

    -- N x (54 + Q) or N x (63 + Q) (encode grid id)
    local data = torch.cat(itemIds, questionIds, 2)
    local labels = answerIds:long()
    if objective == 'regression' then
        labels = labels - synthqa.dict['0']
        labels = labels:float()
    end
    logger:logInfo('encoded labels', 2)
    logger:logInfo(labels, 2)
    return data, labels
end

-------------------------------------------------------------------------------
function synthqa.checkExists(dataset, example)
    for i, example2 in ipairs(dataset) do
        if example.items.question == example2.items.question and 
            example.items.answer == example2.items.answer then
            local allSame = true
            for j, item2 in ipairs(example2.items) do
                local item = example.items[j]
                -- print(item)
                if item2.category ~= item.category or
                    item2.color ~= item.color or
                    item2.grid ~= item.grid then
                    allSame = false
                    break
                end
            end
            if allSame then
                return true
            end
        end
    end
    return false
end

-------------------------------------------------------------------------------
function synthqa.checkOverlap(dataset1, dataset2)
    local overlap = 0
    for i, example in ipairs(dataset2) do
        if synthqa.checkExists(dataset1, example) then
            overlap = overlap + 1
        end
    end
    logger:logInfo(string.format(
        '%.3f of dataset2 overlap with dataset1', overlap / #dataset2))
end

-------------------------------------------------------------------------------

-- local rawData = synthqa.genHowManyObject(5000)
-- local rawData2 = synthqa.genHowManyObject(5000)
-- local data, labels = synthqa.prep(rawData, 'regression')
-- synthqa.checkOverlap(rawData, rawData2)
-- nnevaluator.getClassAccuracyAnalyzer(
--     function(x) return x end, 
--     {'0','1','2','3','4','5','6','7','8','9'},
--     0)(labels, labels)

return synthqa
