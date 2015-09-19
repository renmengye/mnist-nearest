local torch = require('torch')
local logger = require('logger')()
local tableUtils = require('table_utils')
local imageqa = require('imageqa')
local synthqa = {}

torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------

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

synthqa.numObj1 = {
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine'
}

synthqa.NUM_GRID = #synthqa.X * #synthqa.Y
synthqa.OBJECT_SIZE_AVG = 0.1
synthqa.OBJECT_SIZE_STD = 0.03
synthqa.X_STD = 0.05
synthqa.Y_STD = 0.05

----------------------------------------------------------------------

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

local imageSize = {300, 300}

----------------------------------------------------------------------

function synthqa.genHowManyObject(N)
    -- Uniform distribution between 0 and 9
    local dataset = {}
    for i = 1, N do
        local numObj1 = torch.floor(torch.uniform() * #synthqa.numObj1)
        local objectOfInterest = i % #synthqa.OBJECT + 1
        local question = string.format(
            'how many %s', synthqa.OBJECT[objectOfInterest])
        local answer = synthqa.numObj1[numObj1 + 1]
        -- We shuffle a list of 1-9
        -- The first #numObj1 of grids are object of interest
        -- Then we have some numObj1 of other objects
        -- Then we have empty grids
        local sampleGrid = torch.randperm(synthqa.NUM_GRID)
        local items = {}
        if numObj1 > 0 then
            for j = 1, numObj1 do
                table.insert(items, {
                    category = objectOfInterest, 
                    grid = sampleGrid[j]
                })
            end
        end
        local numObj2 = torch.floor(
            torch.uniform() * (synthqa.NUM_GRID - numObj1))
        local nextObj = torch.ceil(torch.uniform() * 2)
        if numObj2 > 0 then
            for j = numObj1 + 1, numObj1 + numObj2 do
                table.insert(items, {
                    category = (objectOfInterest - 1 + nextObj) % 
                               #synthqa.OBJECT + 1,
                    grid = j 
                })
            end
        end
        local numObj3 = torch.floor(
            torch.uniform() * (synthqa.NUM_GRID - numObj1 - numObj2))
        if numObj3 > 0 then
            for j = numObj1 + numObj2 + 1, numObj1 + numObj2 + numObj3 do
                table.insert(items, {
                    category = (objectOfInterest - 1 + 2 * nextObj) % 
                               #synthqa.OBJECT + 1,
                    grid = j
                })
            end
        end
        logger:logInfo(
            string.format(
                'N1: %d, N2: %d, N3: %d, N: %d', 
                numObj1, numObj2, numObj3, numObj1 + numObj2 + numObj3), 2)
        logger:logInfo(
            string.format(
                'Q: %s, A: %s', 
                question, answer), 2)
        logger:logInfo(table.tostring(items))
        table.insert(dataset, {
            question = question,
            answer = answer,
            items = items
        })
    end
    return dataset
end

----------------------------------------------------------------------

function synthqa.encodeItems(items)
    -- Category ID (1)
    -- Color ID (1)
    -- X, Y coordinates (2)
    -- local colorIdict = imageqa.invertDict(COLOR)
    -- local objTypeIdict = imageqa.invertDict(OBJECT)
    local function getCoord(grid)
        -- Randomly sample with gaussian noise
        local yCenter = torch.floor((grid - 1) / 3)  / 3 + (1 / 6)
        local xCenter = (grid - 1) % 3 / 3 + (1 / 6)
        logger:logInfo(string.format('Grid %d, X center %f Y center %f', grid, xCenter, yCenter), 2)
        return torch.Tensor({torch.normal(xCenter, synthqa.X_STD),
                torch.normal(yCenter, synthqa.Y_STD),
                torch.normal(synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD),
                torch.normal(synthqa.OBJECT_SIZE_AVG, synthqa.OBJECT_SIZE_STD)})
    end
    local numDim = 1 + 1 + 4
    local result = torch.Tensor(#items, synthqa.NUM_GRID * numDim):zero()
    for i, example in ipairs(items) do
        local itemShuffle = torch.randperm(#example)
        for j = 1, #example do
            local item = example[itemShuffle[j]]
            for key, value in pairs(item) do
                if key == 'category' then
                    result[{i, (j - 1) * numDim + 1}] = value
                elseif key == 'color' then
                    result[{i, (j - 1) * numDim + 2}] = value
                elseif key == 'grid' then
                    result[{i, {(j - 1) * numDim + 3, (j - 1) * numDim + 6}}] = getCoord(value)
                end
            end
        end
    end
    return result
end

----------------------------------------------------------------------

function synthqa.prep(rawData)
    local dict = combine({
            synthqa.OBJECT, 
            synthqa.X, 
            synthqa.Y, 
            synthqa.COLOR, 
            synthqa.RELATION, 
            synthqa.WORDS,
            synthqa.numObj1
        })
    local idict = imageqa.invertDict(dict)
    logger:logInfo(table.tostring(dict))
    logger:logInfo(table.tostring(idict))
    local questions = {}
    local answers = {}
    local allItems = {}
    for i, entry in ipairs(rawData) do
        table.insert(questions, entry.question)
        table.insert(answers, entry.answer)
        table.insert(allItems, entry.items)
    end
    local questionIds = imageqa.encodeSentences(questions, dict, true)
    local answerIds = imageqa.encodeSentences(answers, dict, true):reshape(#answers)
    local itemIds = synthqa.encodeItems(allItems)
    logger:logInfo(questionIds, 2)
    logger:logInfo(answerIds, 2)
    logger:logInfo(itemIds:reshape(#answers, synthqa.NUM_GRID, 6), 2)
end

----------------------------------------------------------------------

function synthqa.createModel()
end

----------------------------------------------------------------------

local N = 10
local rawData = synthqa.genHowManyObject(N)
local prepData = synthqa.prep(rawData)
