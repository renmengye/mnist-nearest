local GradientStopper, _ = torch.class('mynn.GradientStopper', 'nn.Module')

function GradientStopper:updateOutput(input)
   self.output = input
   return self.output
end


function GradientStopper:updateGradInput(input, gradOutput)
    if type(input) == 'table' then
        self.gradInput = {}
        for i, v in ipairs(input) do
            self.gradInput[i] = torch.Tensor(v:size()):zero()
        end
    else
        self.gradInput = torch.Tensor(input:size()):zero()
    end
    return self.gradInput
end
