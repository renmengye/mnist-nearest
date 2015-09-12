local Embedding, parent = torch.class('nn.Embedding', 'nn.Module')

function Embedding:__init(inputSize, outputSize, initWeight)
    parent.__init(self)
    self.outputSize = outputSize
    if initWeight ~= nil then
        self.weight = initWeight
    else
        self.weight = torch.Tensor(inputSize, outputSize)
    end
    self.gradWeight = torch.Tensor(inputSize, outputSize)
end

function Embedding:updateOutput(input)
    self.output:resize(input:size(1), self.outputSize)
    for i = 1, input:size(1) do
        self.output[i]:copy(self.weight[input[i]])
    end
    return self.output
end

function Embedding:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput:resize(input:size())
        return self.gradInput
    end
end

function Embedding:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if scale == 0 then
        self.gradWeight:zero()
    end
    for i = 1, input:size(1) do
        local word = input[i]
        self.gradWeight[word]:add(gradOutput[i])
    end
end

-- we do not need to accumulate parameters when sharing
Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
