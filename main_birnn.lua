require 'nn'
require 'loadcaffe'
require 'utils.dataloadermat'
require 'models.convrnn'
require 'models.node'
require 'models.tree'
require 'models.renet'
require 'models.renet2'
require 'utils.optim_updates'
local util = require 'utils.util'
local utils = require 'utils.utils'
local net_utils = require 'utils.net_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an VQA model')
cmd:text()
cmd:text('Options')
cmd:option('-cnn_proto','/home/yesenmao/application/eclipse/workspace/orient/model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','/home/yesenmao/application/eclipse/workspace/orient/model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
--./trained models/m2.t7
cmd:option('-input_h5','/home/yesenmao/application/eclipse/workspace/trainmap/data/data_FLIC.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/home/yesenmao/application/eclipse/workspace/trainmap/data/data_FLIC.json','path to the json file containing additional info and vocab')

cmd:option('-batch_size',3,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')

-- Optimization: for the Language Model
cmd:option('-optim','sgd','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 500, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 800, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', './checkpoint', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')


-- Optimization: for the CNN
cmd:option('-finetune_cnn_after', 4000, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

cmd:option('-cnn_optim','sgd','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-4*256,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')


cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 3, 'which gpu to use. -1 = use CPU')   --1 for small GPU
cmd:text()
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
 
local cnn_backend = opt.backend
 if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
 
local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
local cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
--nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]
local fcn6 = nn.SpatialDilatedConvolution(512,1024,7,7,1,1,3,3,1,1)
local relu = nn.ReLU()
local jointskernel1 = nn.ParallelTable()
local jointsbn = nn.ParallelTable()
for i = 1,12 do 
  jointskernel1:add(nn.SpatialConvolution(1024,128,3,3,1,1,1,1))
  jointsbn:add(nn.SpatialBatchNormalization(128,nil,nil,false))
end


local jointskernel2 = nn.ParallelTable()
for i = 1,12 do 
  jointskernel2:add(nn.SpatialConvolution(256,1,1,1,1,1,0,0))
end
--

local weights = torch.Tensor(12):fill(1)
weights[1] = 0.001

local criterion = nn.CrossEntropyCriterion(weights)--12 softmax

-- a fixed tree
--local parents = {2,3,9,10,4,5,8,9,13,13,10,11,14,0}
local parents = {9,1,2,10,4,5,1,4,11,11,0}
--convolutional lstm parameter
local convopt = {}
  convopt.inputPlane = 128
  convopt.outputPlane = 128
  convopt.kW = 3
  convopt.kH = 3
  convopt.dW = 1
  convopt.dH = 1
  convopt.padW = 1
  convopt.padH = 1
  convopt.memorycellsize = {56,56}
  convopt.inputsize = {56,56}
  convopt.batch_size = opt.batch_size
-------------------------------------------------------------------
local nodecore = nn.convLSTM(convopt)
local nodecore_down = nn.convLSTM(convopt)
if opt.gpuid >= 0 then
  cnn:cuda()
  fcn6:cuda()
  relu:cuda()
  jointsbn:cuda()
  jointskernel1:cuda()
  jointskernel2:cuda()
  criterion:cuda()
  nodecore:cuda()
end
local params1, grad_params1 = fcn6:getParameters()
local params2, grad_params2 = jointskernel1:getParameters()
local params3, grad_params3 = jointskernel2:getParameters()
local params4, grad_params4 = nodecore:getParameters()
local cnn_params, cnn_grad_params = cnn:getParameters()

local iter = 1
local function trainloss()
fcn6:training()
relu:training()
cnn:training()
jointsbn:training()
jointskernel1:training()
jointskernel2:training()
nodecore:training()
grad_params1:zero()
grad_params2:zero()
grad_params3:zero()
grad_params4:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end
 local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
 data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0)
 local img = data.images
 local gt = data.gt
 local cnnfeature = cnn:forward(img)
 local fcn6fea = fcn6:forward(cnnfeature)
 local fcn6fearelu = relu:forward(fcn6fea)
 local jointsinput = {}
 for i = 1,12 do
  jointsinput[i] = fcn6fearelu
 end
 local responseMapp = jointskernel1:forward(jointsinput)
 local responseMap = jointsbn:forward(responseMapp)
 local responseMaprelu = {}
 for i = 2,12 do
  responseMaprelu[i-1] = relu:forward(responseMap[i])
 end
 local responseMaprelu1 = relu:forward(responseMap[1])
   --tree lstm
   
  local posetree = util.inputs2tree(responseMaprelu,nodecore,parents)
  posetree:add_core_down(nodecore_down)
  local treemaps_up = posetree:forwardup()
  local treemaps_down = posetree:forwarddown()
  local dim = {gt:size(1),1,56,56}
  local treemaps = {}
  for i = 2,12 do
   treemaps[i] = torch.CudaTensor(dim[1],256,dim[3],dim[4])
   treemaps[i][{{},{1,128},{},{}}] = treemaps_up[i-1]
   treemaps[i][{{},{129,256},{},{}}] = treemaps_down[i-1]
  end
  treemaps[1] = torch.CudaTensor(dim[1],256,dim[3],dim[4])
  treemaps[1][{{},{1,128},{},{}}] = responseMaprelu1 
  treemaps[1][{{},{129,256},{},{}}] = responseMaprelu1 
 -- 1*1 convolution
 local jointmaps = jointskernel2:forward(treemaps)
 local concatMap = torch.CudaTensor(dim[1],dim[2]*12,dim[3],dim[4])
 for i = 1,12 do
  concatMap[{{},{i,i},{},{}}] = jointmaps[i]
 end

 local rr = concatMap:permute(1,3,4,2):contiguous():view(-1,12):contiguous()
 local rrr,r = torch.max(concatMap,2)
  r = r:squeeze()
  print(r[2]:eq(1):sum(),r[2]:eq(2):sum(),r[2]:eq(3):sum(),r[2]:eq(4):sum(),r[2]:eq(5):sum(),r[2]:eq(6):sum()
  ,r[2]:eq(7):sum(),r[2]:eq(8):sum() ,r[2]:eq(9):sum(),r[2]:eq(10):sum(),r[2]:eq(11):sum(),
  r[2]:eq(12):sum())
  print('---------------------------')
 local gr = gt:view(-1,1):contiguous()+1
 gr = gr:cuda()


 local loss = criterion:forward(rr,gr)
 --back propogation
 local d = criterion:backward(rr,gr)
 local dr = d:view(dim[1],dim[3],dim[4],12):contiguous():permute(1,4,2,3):contiguous()
 local dconcatMap = {}
 for i = 1,12 do
  dconcatMap[i] = dr[{{},{i,i},{},{}}]:contiguous() 
 end

 local djointmaps= jointskernel2:backward(treemaps,dconcatMap)
 
 local dtree_up = {}
 local dtree_down = {}
 for i = 2,12 do
  dtree_up[i] = djointmaps[i][{{},{1,128},{},{}}]:contiguous()
  dtree_down[i] = djointmaps[i][{{},{129,256},{},{}}]:contiguous()
  
 end
 
 local dtreemaps_up = posetree:backwardup(dtree_up)
 local dtreemaps_down = posetree:backwarddown(dtree_down)
 local dtreemapss = {}
 for i = 2,12 do
  dtreemapss[i] = dtreemaps_up[i-1]+dtreemaps_down[i-1]
 end
  dtreemapss[1] = djointmaps[1][{{},{1,128},{},{}}]:contiguous()+djointmaps[1][{{},{129,256},{},{}}]:contiguous()
 local dresponseMaprelu = {}
 for i = 1,12 do
   dresponseMaprelu[i] = relu:backward(responseMap[i],dtreemapss[i])
 end
 local dresponseMapp = jointsbn:backward(responseMapp,dresponseMaprelu)
 local dresponseMap = jointskernel1:backward(jointsinput,dresponseMapp)
 local dgradfc6input = dresponseMap[1]:clone()
 for i = 2,12 do
   dgradfc6input = dresponseMap[i]+dgradfc6input
 end
 local dfcn6 = fcn6:backward(cnnfeature,dgradfc6input)
 local losses = { total_loss = loss }
 return losses
end

local function evaluate(split,evalopt)
  fcn6:evaluate()
  relu:evaluate()
  cnn:evaluate()
  jointsbn:evaluate()
  jointskernel1:evaluate()
  jointskernel2:evaluate()
  nodecore:evaluate()
 local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
 local n = 0
 local loss_evals = 0
 local loss = 0
 while true do
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'val', seq_per_img = opt.seq_per_img}
  n = n + data.images:size(1)
  data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0)
  local img = data.images
  local gt = data.gt 
  local cnnfeature = cnn:forward(img)
 local fcn6fea = fcn6:forward(cnnfeature)
 local fcn6fearelu = relu:forward(fcn6fea)
 local jointsinput = {}
 for i = 1,12 do
  jointsinput[i] = fcn6fearelu
 end
 local responseMapp= jointskernel1:forward(jointsinput)
 local responseMap = jointsbn:forward(responseMapp)
 local responseMaprelu = {}
 for i = 2,12 do
  responseMaprelu[i-1] = relu:forward(responseMap[i])
 end
 local responseMaprelu1 = relu:forward(responseMap[1])
   --tree lstm
   
  local posetree = util.inputs2tree(responseMaprelu,nodecore,parents)
  posetree:add_core_down(nodecore_down)
  local treemaps_up = posetree:forwardup()
  local treemaps_down = posetree:forwarddown()
  local treemaps = {}
  for i = 2,12 do
   treemaps[i] = treemapss[i-1]
  end
  treemaps[1] = responseMaprelu1 
 -- 1*1 convolution
 local jointmaps = jointskernel2:forward(treemaps)


 local dim = {gt:size(1),1,56,56}
 local concatMap = torch.CudaTensor(dim[1],dim[2]*12,dim[3],dim[4])
 for i = 1,12 do
  concatMap[{{},{i,i},{},{}}] = jointmaps[i]
 end


   local rr = concatMap:permute(1,3,4,2):contiguous():view(-1,12):contiguous() 
local rrr,r = torch.max(concatMap,2)
  r = r:squeeze()
  print(r[2]:eq(1):sum(),r[2]:eq(2):sum(),r[2]:eq(3):sum()
 ,r[2]:eq(4):sum()
 ,r[2]:eq(5):sum()
 ,r[2]:eq(6):sum()
 ,r[2]:eq(7):sum()
 ,r[2]:eq(8):sum()
 ,r[2]:eq(9):sum()
 ,r[2]:eq(10):sum()
 ,r[2]:eq(11):sum()
 ,r[2]:eq(12):sum())
  print('---------------------------')
   local gr = gt:view(-1,1):contiguous()+1
   gr = gr:cuda()
   -- i am not sure whether this complemention will cause error in data sequence
   loss = criterion:forward(rr,gr) + loss
   loss_evals = loss_evals + 1
   if loss_evals % 5 == 0 then collectgarbage() end
   if data.bounds.wrapped then break end -- the split ran out of data, lets break out
   if n >= val_images_use then break end -- we've used enough images
 end
 local losses = { total_loss = loss }
 return loss/loss_evals
end
 --------------------------------------------------------------------------------
--main loop
local loss0
local optim_state1 = {}
local optim_state2 = {}
local optim_state3 = {}
local optim_state4 = {}
local optim_statek1 = {}
local optim_statek2 = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

while true do  

  -- eval loss/gradient
  local losses = trainloss()
 
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss = evaluate('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
  
    val_loss_history[iter] = val_loss


    --local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)
    local checkpoint_path = './checkpoint/m2'

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    
    checkpoint.val_lang_stats_history = val_lang_stats_history
    
   
    --print('wrote json checkpoint to ' .. checkpoint_path .. '.json')
    utils.write_json(checkpoint_path .. '.json', checkpoint)


    -- write the full model checkpoint as well if we did better than ever

    local current_score = -val_loss
    
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
       -- these are shared clones, and point to correct param storage
  
        save_protos.cnn = cnn
        save_protos.nodecore = nodecore
        save_protos.fcn6 = fcn6
        save_protos.relu = relu
        --save_protos.renet1 = renet1
        --save_protos.renet2 = renet2
        save_protos.jointsbn = jointsbn
        save_protos.jointskernel1 = jointskernel1
        save_protos.jointskernel2 = jointskernel2
   
        checkpoint.protos = save_protos
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params1, grad_params1, opt.learning_rate)
    sgd(params2, grad_params2, opt.learning_rate)
    sgd(params3, grad_params3, opt.learning_rate)
    sgd(params4, grad_params4, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params1, grad_params1, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state1)
    adam(params2, grad_params2, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state2)
    adam(params3, grad_params3, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state3)
    adam(params4, grad_params4, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state4)
   -- adam(params4, grad_params4, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state4)
  else
    error('bad option opt.optim')
  end

  --do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 100 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end


