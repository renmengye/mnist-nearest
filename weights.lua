local torch = require('torch')
local nn = require('nn')
local mynn = require('mynn')

-- A module that outputs constant weights
local Weights, parent = torch.class('mynn.Weights', 'nn.Module')

function Weights:__init(...)
    parent.__init(self)
    self.shape = torch.LongStorage({...})
    self.initRange = 0.1
    self.weight = torch.Tensor(self.shape):uniform(
        -self.initRange / 2, self.initRange / 2)
    self.gradWeight = torch.Tensor(self.weight:size()):zero()
end

function Weights:updateOutput(input)
    -- print('Weights')
    local outputShape = torch.LongStorage(self.shape:size() + 1)
    local reshapeShape = torch.LongStorage(self.shape:size() + 1)
    -- Batch is the same
    outputShape[1] = input:size(1)
    reshapeShape[1] = 1
    for i = 1, self.shape:size() do
        outputShape[i + 1] = self.shape[i]
        reshapeShape[i + 1] = self.shape[i]
    end
    -- print(outputShape)
    return self.weight:reshape(reshapeShape):expand(outputShape)
end

function Weights:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput = torch.Tensor(input:size()):fill(0)
    end
end

function Weights:accGradParameters(input, gradOutput, scale)
    if scale == 0 then
        self.gradWeight:zero()
    end
    self.gradWeight:add(gradOutput:sum(1))
end

-- we do not need to accumulate parameters when sharing
Weights.sharedAccUpdateGradParameters = Weights.accUpdateGradParameters
