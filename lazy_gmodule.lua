local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local utils = require('utils')
local table_utils = require('table_utils')

local LazyGModule, parent = torch.class('nn.LazyGModule', 'nn.gModule')

-------------------------------------------------------------------------------
function LazyGModule:__init(...)
    params = {...}
    parent.__init(self, params[1], params[2])
    self.moduleMap = {}
    self.moduleNames = {}
    self.moduleList = {}
end

-------------------------------------------------------------------------------
function LazyGModule:addModule(name, module)
    self.moduleMap[name] = module.data.module
    table.insert(self.moduleNames, name)
    table.insert(self.moduleList, module.data.module)
end

-------------------------------------------------------------------------------
function LazyGModule:addModuleMap(moduleMap)
    for key, value in pairs(moduleMap) do
        self.moduleMap[key] = value.data.module
        table.insert(self.moduleNames, key)
        table.insert(self.moduleList, value.data.module)
    end
end

-------------------------------------------------------------------------------
function LazyGModule:setup()
    self.w, self.dl_dw = utils.combineAllParameters(self.moduleList)
    self.parameterMap = utils.getParameterMap(self.moduleList, self.moduleNames)
    self.sliceLayer = utils.sliceLayer(self.parameterMap)
    self.isSetup = true
end

-------------------------------------------------------------------------------
function LazyGModule:expand()
    for k, mod in ipairs(self.moduleList) do
        if mod.expand then
            mod:expand()
        end
    end
end

-------------------------------------------------------------------------------
function LazyGModule:getParameters()
    if not self.isSetup then
        self:setup()
    end
    return self.w, self.dl_dw
end

-------------------------------------------------------------------------------
function LazyGModule:parameters()
    return {self.w}, {self.dl_dw}
end
