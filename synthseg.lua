local torch = require('torch')
local table_utils = require('table_utils')
local logger = require('logger')()
local synthseg = {}
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')

-------------------------------------------------------------------------------
synthseg.OBJECT = {
    'A',
    'B',
    'C',
    'E'
}
synthseg.MIN_ITEM_LEN = 2
synthseg.MAX_ITEM_LEN = 4
synthseg.MAX_LEN = 30

-------------------------------------------------------------------------------
function synthseg.gen(N)
    -- Total number of object
    local dataset = {}
    for i = 1, N do
        local objectOfInterest = i % (#synthseg.OBJECT - 1) + 1
        local query = synthseg.OBJECT[objectOfInterest]
        local items = {}
        local j = 0
        while true do
            -- Uniform from 1 - 4
            local objectCat = torch.ceil(torch.uniform() * #synthseg.OBJECT)
            -- Uniform from 2 - 4
            local objectSize = torch.ceil(
                torch.uniform() * (synthseg.MAX_ITEM_LEN - 1)) + 1
            -- logger:logInfo(objectSize)
            j = j + objectSize
            if j > synthseg.MAX_LEN then
                break
            end
            for k = 1, objectSize do
                table.insert(items, {
                    category = objectCat,
                    part = k
                })
            end
        end
        table.insert(dataset, {
            items = items, 
            objectOfInterest = objectOfInterest
        })
    end
    return dataset
end

-------------------------------------------------------------------------------
function synthseg.encodeItems(items)
    -- Encoding length = num objects + max_item_len
    local encodingDim = #synthseg.OBJECT + synthseg.MAX_ITEM_LEN
    local result = torch.Tensor(#items, synthseg.MAX_LEN, encodingDim):zero()
    logger:logInfo(string.format(
        'item encoding dimension: %d', encodingDim))
    for i = 1, #items do
        for j = 1, #items[i] do
            local item = items[i][j]
            result[i][j][item.category] = 1
            result[i][j][#synthseg.OBJECT + item.part] = 1
        end
        if #items[i] < synthseg.MAX_LEN then
            for j = #items[i] + 1, synthseg.MAX_LEN do
                result[i][j][#synthseg.OBJECT] = 1
                result[i][j][#synthseg.OBJECT + 1] = 1
            end
        end
    end
    -- logger:logInfo(result)
    return result
end

-------------------------------------------------------------------------------
function synthseg.getSemanticLabels(dataset)
    local result = torch.Tensor(#dataset, synthseg.MAX_LEN):zero()
    for i = 1, #dataset do
        for j = 1, #dataset[i].items do
            if dataset[i].items[j].category == dataset[i].objectOfInterest then
                result[i][j] = 1
            end
        end
    end
    -- logger:logInfo(result)
    return result
end

-------------------------------------------------------------------------------
function synthseg.encodeQuery(dataset)
    local queryDim = #synthseg.OBJECT
    local result = torch.Tensor(#dataset, queryDim):zero()
    for i = 1, #dataset do
        result[i][dataset[i].objectOfInterest] = 1
    end
    return result
end

-------------------------------------------------------------------------------
function synthseg.getItems(dataset)
    local items = {}
    for i = 1, #dataset do
        table.insert(items, dataset[i].items)
    end
    return items
end

-------------------------------------------------------------------------------
function synthseg.encodeDataset(dataset)
    local items = synthseg.getItems(dataset)
    local itemsEncoded = synthseg.encodeItems(items)
    local itemsEncodedReshape = itemsEncoded:reshape(
        #dataset, itemsEncoded:numel() / #dataset)
    local queryEncoded = synthseg.encodeQuery(dataset)
    return torch.cat(itemsEncodedReshape, queryEncoded, 2)
end

-------------------------------------------------------------------------------
function synthseg.encodeDataset2(dataset)
    -- Encode without query but with semantic GT
    local items = synthseg.getItems(dataset)
    local itemsEncoded = synthseg.encodeItems(items)
    local semLabels = synthseg.getSemanticLabels(dataset)
    semLabels = semLabels:reshape(semLabels:size(1), semLabels:size(2), 1)
    local data = torch.cat(itemsEncoded, semLabels, 3)
    return data
    -- return data:reshape(data:size(1), data:numel() / data:size(1))
end

-------------------------------------------------------------------------------
function synthseg.getOneInstanceLabels(dataset)
    local result = torch.Tensor(#dataset, synthseg.MAX_LEN):zero()
    for i = 1, #dataset do
        local part = 1
        for j = 1, #dataset[i].items do
            if dataset[i].items[j].category == dataset[i].objectOfInterest and
                dataset[i].items[j].part == part then
                result[i][j] = 1
                part = part + 1
            elseif dataset[i].items[j].category == dataset[i].objectOfInterest and
                dataset[i].items[j].part ~= part then
                break
            end
        end
    end
    -- logger:logInfo(result)
    return result
end

-------------------------------------------------------------------------------
local dataset = synthseg.gen(10)
-- logger:logInfo(table.tostring(dataset[1]))
-- synthseg.encodeItems({dataset[1].items})
synthseg.getSemanticLabels(dataset)
synthseg.getOneInstanceLabels(dataset)
local data = synthseg.encodeDataset2(dataset)

return synthseg
