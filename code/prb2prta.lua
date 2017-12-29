--EE569 Homework Assignment #4 (Term Project)
--Date  : December 4, 2016
--Name  : Akash Mukesh Joshi
--USC ID: 4703642421
--email : akashjos@usc.edu
-- Problem 2 Part a

require "nn"
require "optim"
require "image"
require 'gnuplot'

function main()
  -- Fix the random seed for debugging.
  --torch.manualSeed(0)
  
  -- Let us create some 8x8 gray-level images.
  -- Each image belongs to one of two patterns: "x" or "+".
  -- Each image has its pattern on a noisy background.
  -- You don't have to understand this part of code.
  -- In your homework you just load the MNIST data and do preprocessing.
  -- Include Path 
  dataset_train = torch.load('/home/acewalker/Documents/EE569DIP/Homework4/mnist-p1b-train.t7')
  dataset_test = torch.load('/home/acewalker/Documents/EE569DIP/Homework4/mnist-p1b-test.t7')

  do
  local N = 100 -- Creating their own network
  -- In this example, only training set is created.
  -- In your homework, there are training and testing datasets.
  -- You should use the training dataset to train the network,
  -- and use testing dataset to see the true performance of it.
 
  -- Invert Image Dataset 
  trainset = {}
  image_p = dataset_train.data:double() -- Change Dataset to 
  image_n = 255 - dataset_train.data:double()
  image_l = dataset_train.label:double()
  trainset.data = image_p:cat(image_n,1):double() -- double type
  trainset.label = image_l:cat(image_l,1):double() -- double type
   
  --trainset.data = trainset.data1:cat(trainset.data2,1):double()
  --trainset.label = trainset.label:cat(trainset.label,1):double()
 
  testset = {}
  testset.data = dataset_test.data:double() -- Change Dataset to 
  testset.label = dataset_test.label:double() -- double type
  
  for i = 1,10000 do
    local rn = torch.random(0,1)
    if rn > 0 then
      testset.data[{ {i},1,{},{} }] = 255 - testset.data[{ {i},1,{},{} }];
    end
  end
  
  --print(testset.label)
  -- Preview the dataset if you want.
  image.save('Test_Patterns_P2A.png',image.toDisplayTensor(testset.data:narrow(1,1,100),1,10))
  end
  
  --normalize train set globally:
  local mean_u = {} -- store the mean, to normalize the test set in the future
  local std_u  = {} -- store the standard-deviation for the future
  mean_u = trainset.data[{ {},1,{},{} }]:mean()
  std_u = trainset.data[{ {},1,{},{} }]:std()
  trainset.data[{ {},1,{},{} }]:add(-mean_u)
  trainset.data[{ {},1,{},{} }]:div(std_u) 

  -- normalize test set globally:
  mean_u = {} -- store the mean, to normalize the test set in the future
  std_u  = {} -- store the standard-deviation for the future
  mean_u = testset.data[{ {},1,{},{} }]:mean()
  std_u = testset.data[{ {},1,{},{} }]:std()
  testset.data[{ {},1,{},{} }]:add(-mean_u)
  testset.data[{ {},1,{},{} }]:div(std_u) 
  

  -- Create the network and the criterion.
  -- This is just an example which contains all layers you will be using.
  -- However, the architecture and the parameters we should use are different.
  print('Creating the network and the criterion...')
  local network   = nn.Sequential()
  local criterion = nn.ClassNLLCriterion()
  -- 1 input image channels, 6 output channels, 5x5 convolution kernel
  network:add(nn.SpatialConvolution(1, 6, 5, 5)) 
  -- non-linearity 
  network:add(nn.ReLU())                       
  -- A max-pooling operation that looks at 2x2 windows and finds the max.
  network:add(nn.SpatialMaxPooling(2,2,2,2))     
  network:add(nn.SpatialConvolution(6, 16, 5, 5))
  -- non-linearity 
  network:add(nn.ReLU())                       
  network:add(nn.SpatialMaxPooling(2,2,2,2))
  -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  network:add(nn.View(16*5*5))                    
  -- fully connected layer (matrix multiplication between input and weights)
  network:add(nn.Linear(16*5*5, 120))             
  -- non-linearity
  network:add(nn.ReLU())                        
  network:add(nn.Linear(120, 84))
  -- non-linearity 
  network:add(nn.ReLU())                       
  -- 10 is the number of outputs of the network (in this case, 10 digits)
  network:add(nn.Linear(84, 10))                   
  -- converts the output to a log-probability. Useful for classification problems
  network:add(nn.LogSoftMax())                     

  -- Extract the parameters and arrange them linearly in memory.
  -- So we have a large vector containing all the parameters.
  parameters,gradParameters = network:getParameters()

  --------------------------------------------------------------------
  -- Total loss.
  local tloss_train = {0,0,0,0,0,0,0,0,0,0}
  local tloss_test = {0,0,0,0,0,0,0,0,0,0}
  local tacc_train = {0,0,0,0,0,0,0,0,0,0}
  local tacc_test = {0,0,0,0,0,0,0,0,0,0}

  local nEpoch = 10
  for e = 1,nEpoch do
    -- Number of training samples.
    local size_train  = trainset.data:size()[1]
    --print('Size of the dataset is',size_train)
    local size_test = testset.data:size()[1]
    -- Batch size. We use a batch of samples to "smooth" the gradients.
    local bsize = 10 -- Increasing Batch Size Increases accuracy
    
    -- Confusion matrix. This is a helpful tool.
    local classes = {'0','1','2','3','4','5','6','7','8','9'} -- {0 to 9}
    local confusion_train = optim.ConfusionMatrix(classes)
    
    -- Specify the training parameters.
    local config = {
      learningRate = 0.06,
    }    

    print('Training...')
    for t  = 1,size_train,10 do
      local bsize = math.min(bsize,size_train-t+1)
      local input  = trainset.data:narrow(1,t,bsize)
      local target = trainset.label:narrow(1,t,bsize)
      
      local function feval()
        -- Reset the gradients to zero.
        gradParameters:zero()
        -- Forward the data and compute the loss.
        local output = network:forward(input)
        local loss   = criterion:forward(output,target)
        -- Collect Statistics
        tloss_train[e] = tloss_train[e] + loss * bsize
        confusion_train:batchAdd(output,target)
        --
        -- Backward. The gradient wrt the parameters are internally computed.
        local gradOutput = criterion:backward(output,target)
        local gradInput  = network:backward(input,gradOutput)
      
        return loss,gradParameters
      end
      --
      -- os.exit()
      -- We use the SGD method.
      optim.sgd(feval, parameters, config)
      -- Show the progress.
      io.write(string.format("progress: %4d/%4d\r",t,size_train))
      io.flush()
    end

    -- Compute the average loss.
    tloss_train[e] = tloss_train[e] / size_train
    -- Update the confusion matrix.
    confusion_train:updateValids()
    -- Let us print the loss and the accuracy.
    -- You should see the loss decreases and the accuracy increases as the training progresses.
    tacc_train[e] = 100*confusion_train.totalValid
    print(string.format('epoch = %2d/%2d  loss = %.2f accuracy = %.2f',e,nEpoch,tloss_train[e],tacc_train[e]))
    -- You can print the confusion matrix if you want.
    --print(confusion)

--------------------------------------------------------------------
    -- Confusion matrix. This is a helpful tool.
    local classes = {'0','1','2','3','4','5','6','7','8','9'} -- {0 to 9}
    local confusion_test = optim.ConfusionMatrix(classes)

    -- Forward the data and compute the loss.
    local output = network:forward(testset.data)
    local loss   = criterion:forward(output,testset.label)
    -- Collect Statistics
    confusion_test:batchAdd(output,testset.label)
    
    -- Compute the average loss.
    tloss_test[e] = loss
    -- Update the confusion matrix.
    confusion_test:updateValids()
    -- Let us print the loss and the accuracy.
    -- You should see the loss decreases and the accuracy increases as the training progresses.
    tacc_test[e] = 100*confusion_test.totalValid
    
    if  e > 9  then
      print('Testing...')
      print(string.format('epoch = %2d/%2d  loss = %.2f accuracy = %.2f',e,nEpoch,tloss_test[e],tacc_test[e]))
      -- You can print the confusion matrix if you want.
      --print(confusion)
    end 
  end
  
  -- Clean temporary data to reduce the size of the network file.
  network:clearState()
  -- Save the network.
  torch.save('output.t7',network)

  -- Plotting Epoch Accuracy on the same graph
  gnuplot.pngfigure('EpochAccuracy.png')
  gnuplot.plot(
    {'Train', torch.Tensor(tacc_train), '-'},
    {'Test', torch.Tensor(tacc_test), '-'})
  gnuplot.xlabel('Epoch Iteration')
  gnuplot.ylabel('Accuracy')
  gnuplot.plotflush()

  print('The Mean Average Precision on the Testing Dataset = %.2f',tacc_test[10])
end

main()