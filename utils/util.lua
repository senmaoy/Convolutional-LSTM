local util ={}
--------------------------------------------------------------------------------
--tree opration

function util.count_table_keys(t,key)
  local c = {}
   for k,v in pairs(t) do
    if v == key then
     table.insert(c,k)
   end
  end
  return c
end
function util.inputs2nodes(inputs,core,parents)
 local opt = {} 
 opt.core = core
 local nodes = {}
 local root

 for k = 1,#parents do
  opt.input = inputs[k]
  opt.parent = parents[k]
  opt.idx = k
  opt.children = util.count_table_keys(parents,k)
  local convnode = node(opt)
  table.insert(nodes,convnode)
  if parents[k] == 0 then
   root = convnode
  end
 end
return nodes,root
end
function util.inputs2tree(inputs,core,parents)
 local nodes,root = util.inputs2nodes(inputs,core,parents)
 local posetree = tree()
 posetree:addNodes(nodes)
 posetree:addroot(root)
 util.allocateChildren(posetree)
 util.allocateParents(posetree)
return posetree
end
function util.allocateChildren(tree)
 local nodes = tree.nodes
 for k,v in pairs(nodes) do
  for kk,vv in pairs(v.children) do
   table.insert(v.childrenP,nodes[v.children[kk]]) 
  end
 end
end

function util.allocateParents(tree)
 local nodes = tree.nodes
 for k,v in pairs(nodes) do
   v.parentP = nodes[v.parent]
 end
end
----------------------------------------------------------------------------------

function util._createInitState(batch_size,rnn_size,num_layers)
  local init_state 
  rnn_size = rnn_size or 512
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not init_state then init_state = {} end -- lazy init
  for h=1,num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if init_state[h] then
      if init_state[h]:size(1) ~= batch_size then
        init_state[h]:resize(batch_size, rnn_size):zero() -- expand the memory
      end
    else
      init_state[h] = torch.zeros(batch_size, rnn_size):cuda()
    end
  end
  return init_state
end



function util.tree2sequnce(parents)
  local tree = util.inputs2tree({},nil,parents)
  local sequence = {}
  local extra = {}
  tree.root:tree2sequence(sequence,extra)
  return sequence,extra
end

return util