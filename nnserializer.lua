local hdf5 = require('hdf5')

local nnserializer = {}

function nnserializer.load(model, filename)
    local f = hdf5.open(filename, 'r')
    local w = model.w
    local dl_dw = model.dl_dw
    local parameterMap = model.parameterMap
    for key, value in pairs(parameterMap) do
        w[{{value[1], value[2]}}]:copy(f:read(key):all())
    end
end

function nnserializer.save(model, filename)
    local f = hdf5.open(filename, 'w')
    local w = model.w
    local dl_dw = model.dl_dw
    if w == nil or dl_dw == nil then
        w, dl_dw = model:getParameters()
    end
    local parameterMap = model.parameterMap
    if parameterMap == nil then
        f:write('__all__', w)
    else
        for key, value in pairs(parameterMap) do
            f:write(key, w[{{value[1], value[2]}}])
        end
    end
    f:close()
end

return nnserializer
