--[[

  A basic node structure.

--]]

local node = torch.class('node')

function node:__init(arg)
  self.parent = nil or arg.parent
  self.parentP = nil --parent pointer
  self.children = arg.children or {}
  self.childrenP = {}
  self.num_children = #self.children
  self.idx = arg.idx
  self.input = arg.input
  self.gradOutput = nil
  
  if arg.core ~= nil then
   self.core = arg.core:clone('weight', 'bias', 'gradWeight', 'gradBias') 
   end
end

function node:forwardup()
  for i = 1, #self.children do self.childrenP[i]:forwardup() end
  if #self.children>0 then
   local child_h,child_c = self:getchildrenstate() 
   self.output = self.core:forward{self.input,child_h,child_c}
  else
   self.output = self.core:forward(self.input)
  end

end

function node:backwardup()
  if self.parent ~= 0 then
   local parent_dh,parent_dc = self:getparentgradient()
   self.gradInput = self.core:backward(nil,{parent_dh+self.gradOutput,parent_dc},1) 
  else
   self.gradInput = self.core:backward(nil,self.gradOutput,1)
  end
  for i = 1, #self.children do self.childrenP[i]:backwardup() end

end

function node:forwarddown()
  if self.parent ~= 0 then
   self.core_down:forward{self.input,unpack(self.parent.core_down.output)}
  else
   self.core_down:forward(self.input)
  end
  for i = 1, #self.children do self.childrenP[i]:forwarddown() end

end
function node:backwarddown()
  for i = 1, #self.children do self.childrenP[i]:backwarddown() end
  if #self.children>0 then
   local child_dh,child_dc = self.getChildrenGradientState() 
   self.core:backward(nil,{child_dh+self.gradOutput,child_dc},1)
  else
   self.core:backward(nil,self.gradOutput,1)
  end
end
function node:getchildrenstate()
  local childrenoutput = {}
  for i = 1, #self.children do
   local o = self.childrenP[i].core.output
   table.insert(childrenoutput,o)
  end
  local sh = childrenoutput[1][1]:clone()
  local sc = childrenoutput[1][2]:clone()
  for i = 2, #self.children do 
   sh = sh+childrenoutput[i][1]
   sc = sc+childrenoutput[i][2]
  end
  return sh,sc
end

function node:getChildrenGradientState()
  local childrenGradInput = {}
  for i = 1, #self.children do
   local g = self.childrenP[i].core_down.gradInput
   table.insert(childrenGradInput,g)
  end
  local dh = childrenGradInput[1][2]:clone()
  local dc = childrenGradInput[1][3]:clone()
  for i = 2, #self.children do 
   dh = dh+childrenGradInput[i][2]
   dc = dc+childrenGradInput[i][3]
  end
  return {dh,dc}
end
function node:getparentgradient()
  local grad = self.parentP.gradInput
  local dh = grad[2]
  local dc = grad[3]
  return dh,dc
end

function node:tree2sequence(sequence,extra)
    for i = 1, #self.children do self.childrenP[i]:tree2sequence(sequence,extra) end
    table.insert(sequence,node.idx)
    table.insert(extra,#self.children)
end