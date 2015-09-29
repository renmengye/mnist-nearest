local torch = require('torch')
local OneHot, parent = torch.class('mynn.OneHot', 'nn.Module')

function OneHot:__init(nInput)
    parent.__init(self)
    self.nInput = nInput
end

function OneHot:updateOutput(input)
    input = input:reshape(input:numel())
    self.output = torch.Tensor(input:numel(), self.nInput):zero()
    for i = 1, input:numel() do
        self.output[i][input[i]] = 1.0
    end
    return self.output
end

function OneHot:updateGradInput(input, gradOutput)
    return torch.Tensor(input:size()):zero()
end
