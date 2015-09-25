local torch = require('torch')
local nn = require('nn')
local utils = require('utils')

local LazySequential, parent = torch.class('nn.LazySequential','nn.Sequential')

-------------------------------------------------------------------------------
function LazySequential:__init()
    parent.__init(self)
    self.moduleMap = {}
    self.moduleNames = {}
    self.moduleList = {}
end

-------------------------------------------------------------------------------
function LazySequential:addModule(name, module)
    self:add(module)
    self.moduleMap[name] = module
    table.insert(self.moduleNames, name)
    table.insert(self.moduleList, module.data.module)
end

-------------------------------------------------------------------------------
function LazySequential:setup()
    self.w, self.dl_dw = utils.combineAllParameters(self.modules)
    self.parameterMap = utils.getParameterMap(self.moduleList, self.moduleNames)
    self.sliceLayer = utils.sliceLayer(self.parameterMap)
    self.isSetup = true
end

-------------------------------------------------------------------------------
function LazySequential:getParameters()
    if not self.isSetup then
        self:setup()
    end
    return self.w, self.dl_dw
end
