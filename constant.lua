local torch = require('torch')
local nn = require('nn')
local mynn = require('mynn')

-- A module that outputs constant weights
local Constant, parent = torch.class('mynn.Constant', 'nn.Module')

function Constant:__init(shape, value)
    parent.__init(self)
    if value == nil then
        self.value = 0
    else
        self.value = value
    end
    if type(shape) == 'number' then
        self.shape = torch.LongStorage({shape})
    else
        self.shape = torch.LongStorage(shape)
    end
end

function Constant:updateOutput(input)
    local outputShape = torch.LongStorage(self.shape:size() + 1)
    -- Batch is the same
    while type(input) == 'table' do
        input = input[1]
    end
    outputShape[1] = input:size(1)
    for i = 1, self.shape:size() do
        outputShape[i + 1] = self.shape[i]
    end
    -- print('Constant')
    -- print(outputShape)
    if self.name == 'ones' then
        print(torch.Tensor(outputShape):fill(self.value))
    end
    return torch.Tensor(outputShape):fill(self.value)
end

function Constant:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput = torch.Tensor(input:size()):fill(0)
    end
end

function Constant:accGradParameters(input, gradOutput, scale)
    if scale == 0 then
        self.gradWeight:zero()
    end
end

-- we do not need to accumulate parameters when sharing
Constant.sharedAccUpdateGradParameters = Constant.accUpdateGradParameters
