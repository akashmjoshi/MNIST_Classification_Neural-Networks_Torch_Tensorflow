--EE569 Homework Assignment #4 (Term Project)
--Date  : December 4, 2016
--Name  : Akash Mukesh Joshi
--USC ID: 4703642421
--email : akashjos@usc.edu
-- Problem 2 Part b

require "nn"
require "optim"
require "image"
require 'gnuplot'

function main()
  -- Fix the random seed for debugging.
  torch.manualSeed(0)
  
  -- Let us create some 8x8 gray-level images.
  -- Each image belongs to one of two patterns: "x" or "+".
  -- Each image has its pattern on a noisy background.
  -- You don't have to understand this part of code.
  -- In your homework you just load the MNIST data and do preprocessing.
  -- Include Path 
  dataset_train = torch.load('/home/acewalker/Documents/EE569DIP/Homework4/mnist-p2b-train.t7')
  dataset_test = torch.load('/home/acewalker/Documents/EE569DIP/Homework4/mnist-p2b-test.t7')

  do
  local N = 100 -- Creating their own network
  -- In this example, only training set is created.
  -- In your homework, there are training and testing datasets.
  -- You should use the training dataset to train the network,
  -- and use testing dataset to see the true performance of it.

  trainset = {}
  trainset.data = dataset_train.data:double() -- Change Dataset to 
  trainset.label = dataset_train.label:double() -- double type
 
  testset = {}
  testset.data = dataset_test.data:double() -- Change Dataset to 
  testset.label = dataset_test.label:double() -- double type

  -- Preview the dataset if you want.
  image.save('Test_Patterns_P2B.png',image.toDisplayTensor(testset.data:narrow(1,1,100),1,10))
  end
  
  -- normalize train set globally:
  local mean_u = {} -- store the mean, to normalize the test set in the future
  local std_u  = {} -- store the standard-deviation for the future
  for i=1,3 do -- over each image channel
    mean_u = trainset.data[{ {},{i},{},{} }]:mean()
    std_u = trainset.data[{ {},{i},{},{} }]:std()
    trainset.data[{ {},{i},{},{} }]:add(-mean_u)
    trainset.data[{ {},{i},{},{} }]:div(std_u) 
  end

  -- normalize test set globally:
  mean_u = {} -- store the mean, to normalize the test set in the future
  std_u  = {} -- store the standard-deviation for the future
  for i=1,3 do -- over each image channel
    mean_u = testset.data[{ {},{i},{},{} }]:mean()
    std_u = testset.data[{ {},{i},{},{} }]:std()
    testset.data[{ {},{i},{},{} }]:add(-mean_u)
    testset.data[{ {},{i},{},{} }]:div(std_u) 
  end

  -- Create the network and the criterion.
  -- This is just an example which contains all layers you will be using.
  -- However, the architecture and the parameters we should use are different.
  print('Creating the network and the criterion...')
  local network   = nn.Sequential()
  local criterion = nn.ClassNLLCriterion()
  -- 3 input image channels, 6 output channels, 5x5 convolution kernel
  network:add(nn.SpatialConvolution(3, 6, 5, 5)) 
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
  -- 10 is the number of outputs of the network (in this case, 10 digits + Background)
  network:add(nn.Linear(84, 11))                   
  -- converts the output to a log-probability. Useful for classification problems
  network:add(nn.LogSoftMax())                     

  -- Extract the parameters and arrange them linearly in memory.
  -- So we have a large vector containing all the parameters.
  parameters,gradParameters = network:getParameters()

  --------------------------------------------------------------------
  -- Total loss.
  local tloss_train = {0,0,0,0,0,0,0,0,0,0}
  local tloss_test = 0
  local tacc_train = {0,0,0,0,0,0,0,0,0,0}
  local tacc_test = 0
  -- Number of training samples.
  local size_train  = trainset.data:size()[1]
  local size_test = testset.data:size()[1]
  
  -- Batch size. We use a batch of samples to "smooth" the gradients.
  local bsize = 10 -- Increasing Batch Size Increases accuracy

  -- Confusion matrix. This is a helpful tool.
  local classes = {'0','1','2','3','4','5','6','7','8','9','B'} -- {0 to 9}
  local confusion = optim.ConfusionMatrix(classes)

  local nEpoch = 10
  for e = 1,nEpoch do 
    
    print('Training...')
    for t  = 1,size_train,10 do
      local bsize = math.min(bsize,size_train-t+1)
      local input  = trainset.data:narrow(1,t,bsize)
      local target = trainset.label:narrow(1,t,bsize)
      -- Reset the gradients to zero.
      gradParameters:zero()
      -- Forward the data and compute the loss.
      local output = network:forward(input)
      local loss   = criterion:forward(output,target)
      -- Collect Statistics
      tloss_train[e] = tloss_train[e] + loss * bsize
      confusion:batchAdd(output,target)
      --
      -- Backward. The gradient wrt the parameters are internally computed.
      local gradOutput = criterion:backward(output,target)
      local gradInput  = network:backward(input,gradOutput)
      -- The optim module accepts a function for evaluation.
      -- For simplicity I made the computation outside, and
      -- this function is used only to return the result.
      local function feval()
        return loss,gradParameters
      end
      -- Specify the training parameters.
      local config = {
        learningRate = 0.06,
      }
      --
      -- os.exit()
      -- We use the SGD method.
      --optim.sgd(feval, parameters, config)
      optim.sgd(feval, parameters, config)
      -- Show the progress.
      io.write(string.format("progress: %4d/%4d\r",t,size_train))
      io.flush()
    end

    -- Compute the average loss.
    tloss_train[e] = tloss_train[e] / size_train
    -- Update the confusion matrix.
    confusion:updateValids()
    -- Let us print the loss and the accuracy.
    -- You should see the loss decreases and the accuracy increases as the training progresses.
    tacc_train[e] = 100*confusion.totalValid
    print(string.format('epoch = %2d/%2d  loss = %.2f accuracy = %.2f',e,nEpoch,tloss_train[e],tacc_train[e]))
    -- You can print the confusion matrix if you want.
    --print(confusion)
  end
-------------------------------------------------------------------
  print('Testing...')
  for t  = 1,size_test,10 do
    local bsize = math.min(bsize,size_test-t+1)
    local input  = testset.data:narrow(1,t,bsize)
    local target = testset.label:narrow(1,t,bsize)
    -- Reset the gradients to zero.
    gradParameters:zero()
    -- Forward the data and compute the loss.
    local output = network:forward(input)
    local loss   = criterion:forward(output,target)
    -- Collect Statistics
    tloss_test = tloss_test + loss * bsize
    confusion:batchAdd(output,target)

    -- Show the progress.
    io.write(string.format("progress: %4d/%4d\r",t,size_test))
    io.flush()
  end
    
  -- Compute the average loss.
  tloss_test = tloss_test / size_test
  -- Update the confusion matrix.
  confusion:updateValids()
  -- Let us print the loss and the accuracy.
  -- You should see the loss decreases and the accuracy increases as the training progresses.
  tacc_test = 100*confusion.totalValid
  print(string.format('loss = %.2f accuracy = %.2f',tloss_test,tacc_test))
  -- You can print the confusion matrix if you want.
  --print(confusion)
------------------------------------------------------------------------
  -- Clean temporary data to reduce the size of the network file.
  network:clearState()
  -- Save the network.
  torch.save('output.t7',network)

  -- Plotting Epoch Accuracy on the same graph
  --gnuplot.pngfigure('EpochAccuracy.png')
  --gnuplot.plot(
  --  {'Train', torch.Tensor(tacc_train), '-'},
  --  {'Test', torch.Tensor(tacc_test), '-'})
  --gnuplot.xlabel('Epoch Iteration')
  --gnuplot.ylabel('Accuracy')
  --gnuplot.plotflush()

  print('The Mean Average Precision on the RGB Testing Dataset = %.2f',tacc_test)
end

main()