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
    local inputShape = input:size()
    local outputShape = torch.LongStorage(inputShape:size() + 1)
    for i = 1, inputShape:size() do
        outputShape[i] = inputShape[i]
    end
    outputShape[inputShape:size() + 1] = self.outputSize
    input = input:reshape(input:numel())
    self.output:resize(input:numel(), self.outputSize)
    for i = 1, input:numel() do
        -- A little hack for now
        if input[i] > 0 then
            self.output[i]:copy(self.weight[input[i]])
        else
            self.output[i]:copy(torch.Tensor(self.outputSize):zero())
        end
    end
    self.output = self.output:reshape(outputShape)
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
    input = input:reshape(input:numel())
    gradOutput = gradOutput:reshape(input:numel(), self.outputSize)
    for i = 1, input:numel() do
        if input[i] > 0 then
            local word = input[i]
            self.gradWeight[word]:add(gradOutput[i])
        end
    end
end

-- we do not need to accumulate parameters when sharing
Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
