local tree = torch.class('tree')

function tree:__init()
 self.nodes = {}
 self.root = nil
 
end

function tree:addNodes(nodes)
 for k,v in pairs(nodes) do table.insert(self.nodes, v) end
end

function tree:addroot(node)
 self.root = node
end

function tree:forwardup()
 self.root:forwardup()
 local output = {}
 for k,v in pairs(self.nodes) do table.insert(output, v.output[1]) end
 return output
end

function tree:backwardup(gradOutput)
 for k,v in pairs(self.nodes) do v.gradOutput = gradOutput[k] end
 self.root:backwardup()
 local gradInput = {}
 for k,v in pairs(self.nodes) do table.insert(gradInput, v.gradInput[1]) end
 return gradInput
end

function tree:forwarddown()
 self.root:forwarddown()
 local output = {}
 for k,v in pairs(self.nodes) do table.insert(output, v.core_down.output[1]) end
 return output
end

function tree:backwarddown(gradOutput)
 for k,v in pairs(self.nodes) do v.gradOutput = gradOutput[k] end
 self.root:backwardup()
 local gradInput = {}
 for k,v in pairs(self.nodes) do table.insert(gradInput, v.core_down.gradInput[1]) end
 return gradInput
end
function tree:add_core_down(core_down)
 for k,v in pairs(self.nodes) do v.core_down = core_down:clone('weight', 'bias', 'gradWeight', 'gradBias')  end
end