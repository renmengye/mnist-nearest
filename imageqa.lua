local knn = require('nearest_neighbours')
local hdf5 = require('hdf5')
local dataPath = '/ais/gobi3/u/mren/data/cocoqa-nearest/all.h5'
local data = hdf5.open(dataPath, 'r'):all()
for key,value in pairs(data) do 
    print(key, value:size())
    collectgarbage()
end
