local GradientStopper, _ = torch.class('nn.GradientStopper', 'nn.Module')

function GradientStopper:updateOutput(input)
   self.output = input
   return self.output
end


function GradientStopper:updateGradInput(input, gradOutput)
   self.gradInput = torch.Tensor(input:size()):zero()
   return self.gradInput
end
