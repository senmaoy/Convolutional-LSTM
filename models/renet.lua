
require 'models.LSTM'
require 'models.LSTMB'


local layer, parent = torch.class('nn.renet', 'nn.Module')

function layer:__init(input_dim,output_dim)
  self.input_dim =input_dim
  self.output_dim = output_dim
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
  self.lstm11 = nn.LSTM(input_dim,output_dim)
  self.lstm12 = nn.LSTMB(input_dim,output_dim)
end


function layer:updateOutput(input)
  local img = input:permute(1,3,4,2):contiguous()
  local feachannal = self.output_dim
  self.output:resize(img:size(1),img:size(2),img:size(3),feachannal*2)
  local out1 = self.output
  local w = img:size(2)
  local h = img:size(3)
  -- the first dimension
  for i = 1,w do
    local seq = img[{{},{i,i},{},{}}]:contiguous():squeeze()
    local outseq1 = self.lstm11:forward(seq)
    local outseq2 = self.lstm12:forward(seq)
    out1[{{},{i,i},{},{1,feachannal}}] = outseq1
    out1[{{},{i,i},{},{feachannal+1,-1}}] = outseq2
  end
  -- the second dimension
  return self.output:permute(1,4,2,3):contiguous()

end


function layer:updateGradInput(input, gradOutput)
  local img = input:permute(1,3,4,2):contiguous()
  local w = img:size(2)
  local h = img:size(3)
  local grad = gradOutput--:permute(1,3,4,2):contiguous()
  local dimg = img:clone()
  local grad21 = grad[{{},{},{},{1,self.output_dim}}]:contiguous()
  local grad22 = grad[{{},{},{},{self.output_dim+1,self.output_dim*2}}]:contiguous()
  for i = 1,w do
    local seq = img[{{},{i,i},{},{}}]:contiguous():squeeze()
    local outseq1 = self.lstm11:backward(seq,grad21[{{},{i,i},{},{}}]:contiguous():squeeze())
    local outseq2 = self.lstm12:backward(seq,grad22[{{},{i,i},{},{}}]:contiguous():squeeze())
    dimg[{{},{i,i},{},{}}] = outseq1+outseq2
  end
  self.gradInput = dimg:permute(1,4,2,3):contiguous()
  return self.gradInput

end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.lstm11:parameters()
  local p2,g2 = self.lstm12:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  return params, grad_params
end
--collect parameters
function layer:training()
  self.lstm11:training()
  self.lstm12:training()
end

function layer:evaluate()
  self.lstm11:evaluate()
  self.lstm12:evaluate()
end