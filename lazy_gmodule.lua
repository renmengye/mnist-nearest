local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local utils = require('utils')
local table_utils = require('table_utils')

local LazyGModule, parent = torch.class('nn.LazyGModule', 'nn.gModule')

-- Currently not working
-------------------------------------------------------------------------------
function LazyGModule:__init(...)
    params = {...}
    print(params[1])
    print(params[1][1])
    parent.__init(self, params[1], params[2])
    self.moduleMap = {}
    self.moduleList = {}
end

-------------------------------------------------------------------------------
function LazyGModule:addModule(name, module)
    self.moduleMap[name] = module.data.module
    table.insert(self.moduleList, module.data.module)
end

-------------------------------------------------------------------------------
function LazyGModule:setup()
    self.w, self.dl_dw = utils.combineAllParameters(self.moduleList)
    self.parameterMap = utils.getParameterMap(self.moduleMap)
    self.sliceLayer = utils.sliceLayer(self.parameterMap)
    self.isSetup = true
end

-------------------------------------------------------------------------------
function LazyGModule:getParameters()
    if not self.isSetup then
        self:setup()
    end
    return self.w, self.dl_dw
end
