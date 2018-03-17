
local THNN = require 'nn.THNN'
local layer, parent = torch.class('nn.convLSTM', 'nn.Module')


function layer:__init(opt)
  parent.__init(self)
  self.inputPlane = opt.inputPlane
  self.outputPlane = opt.outputPlane
  
  self.kW = opt.kW
  self.kH = opt.kH
  
  self.dW = opt.dW
  self.dH = opt.dH
  
  self.padW = opt.padW
  self.padH = opt.padH
  
  self.wxlength = self.outputPlane*self.inputPlane*self.kH*self.kW*4
  self.wclength = self.outputPlane*self.outputPlane*self.kH*self.kW*4
  
  self.inputsize = opt.inputsize
  self.cellsizetable = opt.memorycellsize
  self.cellsize = self.cellsizetable[1]*self.cellsizetable[2]
  self.clength = self.outputPlane*self.cellsize*3
  self.initlength = self.inputPlane*self.kH*self.kW   --in fact there should be two,but in this experiment they are same
  
  self.weight = torch.Tensor(1,self.wxlength+self.wclength+self.clength)
  self.gradWeight = torch.Tensor(1,self.wxlength+self.wclength+self.clength):fill(0)
  
  self.bias = torch.Tensor(8*self.outputPlane)
  self.gradBias = torch.Tensor(8*self.outputPlane):fill(0)
  self:reset()
  --init hidden state memory cell,input,memory cell gradient
  self.c = torch.Tensor(opt.batch_size,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]):fill(0)
  self.h = torch.Tensor(opt.batch_size,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]):fill(0)
  self.x = torch.Tensor(opt.batch_size,self.inputPlane,self.inputsize[1],self.inputsize[2]):fill(0)
  self.dc = torch.Tensor(opt.batch_size,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]):fill(0)
end


function layer:reset(stdv)
if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.initlength)
   end
  self.bias:uniform(-stdv, stdv)
  self.weight:uniform(-stdv, stdv)
  --self.bias:uniform(-0.001, 0.001)
  --self.weight:uniform(-0.001, 0.001)
  return self
end

local function convforward(opt)
  opt.input.THNN.SpatialConvolutionMM_updateOutput(
      opt.input:cdata(),
      opt.output:cdata(),
      opt.weight:cdata(),
      THNN.optionalTensor(opt.bias),
      opt.finput:cdata(),
      opt.fgradInput:cdata(),
      opt.kW, opt.kH,
      opt.dW, opt.dH,
      opt.padW, opt.padH
   )
end


local function convbackward(opt)
   opt.input.THNN.SpatialConvolutionMM_updateGradInput(
         opt.input:cdata(),
         opt.gradOutput:cdata(),
         opt.gradInput:cdata(),
         opt.weight:cdata(),
         opt.finput:cdata(),
         opt.fgradInput:cdata(),
         opt.kW, opt.kH,
         opt.dW, opt.dH,
         opt.padW, opt.padH
      )
end

local function convacc(opt, scale)
  
   opt.input.THNN.SpatialConvolutionMM_accGradParameters(
      opt.input:cdata(),
      opt.gradOutput:cdata(),
      opt.gradWeight:cdata(),
      THNN.optionalTensor(opt.gradBias),
      opt.finput:cdata(),
      opt.fgradInput:cdata(),
      opt.kW, opt.kH,
      opt.dW, opt.dH,
      opt.padW, opt.padH,
      scale
   )
end
local function convolutionparametersparsing(self,x,w,b)
-----this function should be called when init,change it later.....
    --the output size is determined by input and weight,so there is no need to init earlier
    local opt = {}
    opt.input = x
    opt.output = b.new()
    opt.weight = w
    opt.bias = b
    opt.finput = b.new()
    opt.fgradInput = b.new()
    opt.kW = self.kW--kernelsize[1]
    opt.kH = self.kH
    opt.dW = self.dW
    opt.dH = self.dH
    opt.padW = self.padW
    opt.padH = self.padH
    opt.gradOutput = b.new()
    opt.gradInput = b.new()
    opt.gradWeight = w:clone():fill(0)
    opt.gradBias = b:clone():fill(0)
    -- all the data needed in options are given here
    return opt

end
local function conv(self,x,wxc,bc)
    local optxc = convolutionparametersparsing(self,x,wxc,bc)
    convforward(optxc)
    return optxc.output,optxc
end
local function dconv(...)
    local output={}
    for k,v in ipairs{...}do
      convbackward(v)
      convacc(v,1)
      table.insert(output,v)
    end
    return unpack(output)
end
local function op(a,b)
 --element-wise multiplycation
 assert(a:nElement()==b:nElement())
 if a:dim()~=b:dim() then b:resizeAs(a) end
 return torch.cmul(a,b)
end

local function sigmoid(input)
local output=input.new()
   input.THNN.Sigmoid_updateOutput(
      input:cdata(),
      output:cdata()
   )
   return output
end
local function dsigmoid(input, gradOutput,output)
   local gradInput = gradOutput.new()
   input.THNN.Sigmoid_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      gradInput:cdata(),
      output:cdata()
   )
   return gradInput
end

local function tanh(input)
   local output = input.new()
   input.THNN.Tanh_updateOutput(
      input:cdata(),
      output:cdata()
   )
   return output
end
local function dtanh(input,gradOutput,output )
   local gradInput = input.new()
   input.THNN.Tanh_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      gradInput:cdata(),
      output:cdata()
   )
   return gradInput
end
function layer:updateOutput(input)
 ---------input
 --i should condition on the number of input, so 
 local x,h
 if torch.isTensor(input) then
  x = input
  h = self.h
 else
  x = input[1] or self.x
  h = input[2] or self.h
  self.c = input[3] or self.c
 end
 self.batch_size = x:size(1)
 --transfrom the weights into desired shape
 local wx = self.weight[{{1,1},{1,self.wxlength}}]:resize(4,self.outputPlane,self.inputPlane*self.kW*self.kH)
 local wh = self.weight[{{1,1},{self.wxlength+1,self.wxlength+self.wclength}}]:resize(4,self.outputPlane,self.outputPlane*self.kW*self.kH)
 local wc = self.weight[{{1,1},{self.wxlength+self.wclength+1,-1}}]:resize(3,self.outputPlane,self.cellsize)
 local wxi = wx[{{1,1},{},{}}]:resize(self.outputPlane,self.inputPlane*self.kW*self.kH)
 local wxf = wx[{{2,2},{},{}}]:resize(self.outputPlane,self.inputPlane*self.kW*self.kH)
 local wxc = wx[{{3,3},{},{}}]:resize(self.outputPlane,self.inputPlane*self.kW*self.kH)
 local wxo = wx[{{4,4},{},{}}]:resize(self.outputPlane,self.inputPlane*self.kW*self.kH)
 local whi = wh[{{1,1},{},{}}]:resize(self.outputPlane,self.outputPlane*self.kW*self.kH)
 local whf = wh[{{2,2},{},{}}]:resize(self.outputPlane,self.outputPlane*self.kW*self.kH)
 local whc = wh[{{3,3},{},{}}]:resize(self.outputPlane,self.outputPlane*self.kW*self.kH)
 local who = wh[{{4,4},{},{}}]:resize(self.outputPlane,self.outputPlane*self.kW*self.kH)
 local wci = wc[{{1,1},{},{}}]:resize(self.outputPlane,self.cellsizetable[1],self.cellsizetable[2])
 local wcf = wc[{{2,2},{},{}}]:resize(self.outputPlane,self.cellsizetable[1],self.cellsizetable[2])
 local wco = wc[{{3,3},{},{}}]:resize(self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]) 
 
 local b = self.bias:view(8,self.outputPlane)
 local bxi = b[{{1,1},{}}]:resize(self.outputPlane)
 local bxf = b[{{2,2},{}}]:resize(self.outputPlane)
 local bxc = b[{{3,3},{}}]:resize(self.outputPlane)
 local bxo = b[{{4,4},{}}]:resize(self.outputPlane)
 local bhi = b[{{5,5},{}}]:resize(self.outputPlane)
 local bhf = b[{{6,6},{}}]:resize(self.outputPlane)
 local bhc = b[{{7,7},{}}]:resize(self.outputPlane)
 local bho = b[{{8,8},{}}]:resize(self.outputPlane)
 
 
 local xi,optxi = conv(self,x,wxi,bxi)
 local xf,optxf = conv(self,x,wxf,bxf)
 local xc,optxc = conv(self,x,wxc,bxc)
 local xo,optxo = conv(self,x,wxo,bxo)
 
 --because when checking the gradients the input must be a tensor,  h will be fed when initiation
 local hi,opthi = conv(self,h,whi,bhi)  
 local hf,opthf = conv(self,h,whf,bhf)
 local hc,opthc = conv(self,h,whc,bhc)
 local ho,optho = conv(self,h,who,bho)
 
 self.optxi = optxi
 self.optxf = optxf
 self.optxc = optxc
 self.optxo = optxo
 
 self.opthi = opthi
 self.opthf = opthf
 self.opthc = opthc
 self.optho = optho
 
 wci = wci:resize(1,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]):expand(self.batch_size,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2])
 wcf = wcf:resize(1,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]):expand(self.batch_size,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2])
 wco = wco:resize(1,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2]):expand(self.batch_size,self.outputPlane,self.cellsizetable[1],self.cellsizetable[2])
 
 self.wci = wci
 self.wcf = wcf
 self.wco = wco
 
 local ci = op(self.c,wci)
 local cf = op(self.c,wcf)

 local i = sigmoid(xi+hi+ci)
 local f = sigmoid(xf+hf+cf)

 local xct = tanh(xc+hc)
 self.i = i
 self.f = f

 self.xct = xct
 self.ci = ci
 self.cf = cf

 local cc = op(f,self.c)+op(i,xct)
 local co = op(cc,wco)
 local o = sigmoid(xo+ho+co)
 self.o = o
 self.co = co
 local cct = tanh(cc)
 self.cc = cc
 self.cct = cct 
 self.output = {op(cct,o),cc}
 return self.output
end


function layer:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')
  --gradOuptput:the first is hidden state,the second is memory cell 
  local dc,dh
  if torch.isTensor(gradOutput) then
   dh = gradOutput
   dc = self.dc
  else
   dh = gradOutput[1]
   dc = gradOutput[2] or self.dc
  end
  -------o
  local dso = op(dh,self.cct) 
  local dos = dsigmoid(self.optxo.output+self.optho.output+self.co,dso,self.o)
  self.optxo.gradOutput = dos
  local optxo = dconv(self.optxo)--x
  self.optho.gradOutput = dos
  local optho = dconv(self.optho)--h
  local dco = op(self.wco,dos)   --c
  
  local dcct = op(dh,self.o)
  local dcc = dtanh(self.cc,dcct,self.cct)+dco+dc
  -------i
  local dsi = op(dcc,self.xct)
  local dis = dsigmoid(self.optxi.output+self.opthi.output+self.ci,dsi,self.i)
  self.optxi.gradOutput = dis
  local optxi = dconv(self.optxi)--x
  self.opthi.gradOutput = dis
  local opthi = dconv(self.opthi)--h
  local dci = op(self.wci,dis)   --c
  -------f
  local dsf = op(dcc,self.c)
  local dfs = dsigmoid(self.optxf.output+self.opthf.output+self.cf,dsf,self.f)
  self.optxf.gradOutput = dfs
  local optxf = dconv(self.optxf)--x
  self.opthf.gradOutput = dfs
  local opthf = dconv(self.opthf)--h
  local dcf = op(self.wcf,dfs)   --c  
  -------c
  local dxct= op(dcc,self.i)
  local dxc = dtanh(self.optxc.output+self.opthc.output,dxct,self.xct)
  self.optxc.gradOutput = dxc
  local optxc = dconv(self.optxc)--x
  self.opthc.gradOutput = dxc
  local opthc = dconv(self.opthc)--h
  local dc_c = op(dcc,self.f)    --c_c
  
  local dx = optxi.gradInput+optxf.gradInput+optxc.gradInput+optxo.gradInput
  local dh = opthi.gradInput+opthf.gradInput+opthc.gradInput+optho.gradInput
  local dc = dci+dcf+dc_c
  -----------------------weight and bias gradients---------------------------------------
  --i didn't dive batchsize for gradent check
  local dwx = self.gradWeight[{{1,1},{1,self.wxlength}}]:resize(4,self.outputPlane,self.inputPlane*self.kW*self.kH)
  local dwh = self.gradWeight[{{1,1},{self.wxlength+1,self.wxlength+self.wclength}}]:resize(4,self.outputPlane,self.outputPlane*self.kW*self.kH)
  local dwc = self.gradWeight[{{1,1},{self.wxlength+self.wclength+1,-1}}]:resize(3,self.outputPlane,self.cellsize)
  dwx[{{1,1},{},{}}] = dwx[{{1,1},{},{}}]+optxi.gradWeight:view(1,self.outputPlane,self.inputPlane*self.kW*self.kH)
  dwx[{{2,2},{},{}}] = dwx[{{2,2},{},{}}]+optxf.gradWeight:view(1,self.outputPlane,self.inputPlane*self.kW*self.kH)
  dwx[{{3,3},{},{}}] = dwx[{{3,3},{},{}}]+optxc.gradWeight:view(1,self.outputPlane,self.inputPlane*self.kW*self.kH)
  dwx[{{4,4},{},{}}] = dwx[{{4,4},{},{}}]+optxo.gradWeight:view(1,self.outputPlane,self.inputPlane*self.kW*self.kH)
  dwh[{{1,1},{},{}}] = dwh[{{1,1},{},{}}]+opthi.gradWeight:view(1,self.outputPlane,self.outputPlane*self.kW*self.kH)
  dwh[{{2,2},{},{}}] = dwh[{{2,2},{},{}}]+opthf.gradWeight:view(1,self.outputPlane,self.outputPlane*self.kW*self.kH)
  dwh[{{3,3},{},{}}] = dwh[{{3,3},{},{}}]+opthc.gradWeight:view(1,self.outputPlane,self.outputPlane*self.kW*self.kH)
  dwh[{{4,4},{},{}}] = dwh[{{4,4},{},{}}]+optho.gradWeight:view(1,self.outputPlane,self.outputPlane*self.kW*self.kH)
  dwc[{{1,1},{},{}}] = dwc[{{1,1},{},{}}]+torch.sum(op(dis,self.c),1):view(1,self.outputPlane,-1)
  dwc[{{2,2},{},{}}] = dwc[{{2,2},{},{}}]+torch.sum(op(dfs,self.c),1):view(1,self.outputPlane,-1)
  dwc[{{3,3},{},{}}] = dwc[{{3,3},{},{}}]+torch.sum(op(dos,self.cc),1):view(1,self.outputPlane,-1)
  local db = self.gradBias:view(8,self.outputPlane)
  db[{{1,1},{}}] = db[{{1,1},{}}]+optxi.gradBias:view(1,self.outputPlane)
  db[{{2,2},{}}] = db[{{2,2},{}}]+optxf.gradBias:view(1,self.outputPlane)
  db[{{3,3},{}}] = db[{{3,3},{}}]+optxc.gradBias:view(1,self.outputPlane)
  db[{{4,4},{}}] = db[{{4,4},{}}]+optxo.gradBias:view(1,self.outputPlane)
  db[{{5,5},{}}] = db[{{5,5},{}}]+opthi.gradBias:view(1,self.outputPlane)
  db[{{6,6},{}}] = db[{{6,6},{}}]+opthf.gradBias:view(1,self.outputPlane)
  db[{{7,7},{}}] = db[{{7,7},{}}]+opthc.gradBias:view(1,self.outputPlane)
  db[{{8,8},{}}] = db[{{8,8},{}}]+optho.gradBias:view(1,self.outputPlane)
  self.gradInput = {dx,dh,dc}
  return self.gradInput
end

function layer:clearState()
  
end

function layer:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function layer:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end
