local torch = require('torch')
local nn = require('nn')
local nngraph = require('nngraph')
local utils = require('utils')
local table_utils = require('table_utils')

local ExpandGModule, parent = torch.class('nn.ExpandGModule', 'nn.gModule')

-------------------------------------------------------------------------------
function ExpandGModule:__init(...)
    params = {...}
    parent.__init(self, params[1], params[2])
    self.moduleMap = {}
    self.moduleNames = {}
    self.moduleList = {}
end

-------------------------------------------------------------------------------
function ExpandGModule:addModule(name, module)
    self.moduleMap[name] = module.data.module
    table.insert(self.moduleNames, name)
    table.insert(self.moduleList, module.data.module)
end

-------------------------------------------------------------------------------
function ExpandGModule:expand()
    for k, mod in ipairs(self.moduleList) do
        if mod.expand then
            mod:expand()
        end
    end
end
