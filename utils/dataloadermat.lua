require 'hdf5'
local utils = require 'utils.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
 
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.h5_file)
  
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 4, '/images should be a 4D tensor')
  assert(images_size[3] == images_size[4], 'width and height must match')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.max_image_size = images_size[3]
  print(string.format('read %d images of size %dx%dx%d', self.num_images, 
            self.num_channels, self.max_image_size, self.max_image_size))

  -- separate out indexes for each of the provided splits
 self.split_ix = {}
  self.iterators = {}
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the images
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 1) -- number of sequences to return per image

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 448, 448)
  local gt_batch = torch.FloatTensor(batch_size ,56,56)
  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i=1,batch_size do

    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next
    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- fetch the image from h5
    local img = self.h5_file:read('/images'):partial({ix,ix},{1,self.num_channels},
                            {1,self.max_image_size},{1,self.max_image_size})
    img_batch_raw[i] = img

    -- fetch the sequence labels

    
    local gt = self.h5_file:read('/gt'):partial({ix, ix}, {1,56},{1,56})
    
    gt_batch[{{i,i}}] = gt

    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.info.images[ix].id
    info_struct.file_path = self.info.images[ix].file_path
    table.insert(infos, info_struct)
  end
  local data = {}
  data.images = img_batch_raw
  data.gt = gt_batch:permute(1,3,2):contiguous()
  -- i hope that the labels is stored as tables
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos
  return data
end
