local ReinforceContainer, parent = torch.class('ReinforceContainer', 'nn.Module')

function ReinforceContainer:__init(modules)
    self.myModules = modules
end

function ReinforceContainer:reinforce(reward)
    for i, m in ipairs(self.myModules) do
        m:reinforce(reward)
    end
end

return ReinforceContainer
