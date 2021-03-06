# 11-775 Large-Scale Multimedia Analysis

Peter Schaldenbrand <br/>
Spring 2021<br/>
Homework 1 <br/>
Andrew ID: pschalde <br/>
GitHub ID: pschaldenbrand <br/>
Kaggle ID: pittsburghskeet


## Validation

For experimenting with features and parameters, I used 5-fold cross validation.  80% of the data would be used to train and 20% would be validated.  This is done 5 times with different folds each time and the validation accuracies were averaged and used for comparison.  I used Top-1 accuracy as it is conventional and is used in the leaderboard.

When training my models for non-experimentation, I would use early stopping in the models as a form of regularization.  10% of the training data was used as validation and if the training went 10 epochs without lowering the validation accuracy, the training would stop.

I did not use the test data for validation.  Meaning, I only submitted my results to kaggle after the model was determined to fit well using the 5-fold cross validation described above.  I didn't want my results to be finely tuned to the test data.

## SoundNet

Follow instructions from [SoundNet-tensorflow](https://github.com/eborboihuc/SoundNet-tensorflow).

Clone the repository into this one

```
git clone git@github.com:eborboihuc/SoundNet-tensorflow.git
```

Replace this repository's ```extract_feat.py``` and ```util.py``` with the versions in this repository.  There are a few updates we need to save the files properly.

Download SoundNet's pretrained models per their instructions.

Create a list of audio files we would like to extract features from using ```get_wav_file_list.py``` which creates ```wav_file_list.txt``` which has already been added to this repo.

Extract features from the audio:

```
cd SoundNet-tensorflow
python3.5 extract_feat.py -t ../mp3_file_list.txt -m 9 -x 26 -s -p extract
```

The feature extraction took 16 hours and 15 minutes on my machine.  I did not connect a GPU which led to particularly slow extraction times.  All said and done, the ```SoundNet-tensorflow/output``` now has 134,963 files corresponding to 17 layer extractions for each of the 7939 sound files. Thi stakes up approximately 25GB of disk space.  Each file is named according to the layer and sound file name: ```output/HW[id]tf_fea[layer #].npy```.

This will save layers 9-26 of SoundNet.  The 9th layer corresponds to Conv3 in the SoundNet architecture.  I chose to start extracting features here, since earlier layers likely do not extract semantically interesting features as they will still be quite similar to the input.  Also, the output from the first few layers is dimensionally very large and would be difficult to store this much data.  The 26th layer is the output of the last convolutional layer of SoundNet.  Extracting this many layers is unnecessary, but good to have on hand for experimenting with features later.

## SoundNet Features

The output from each soundnet layer creates features that we can use for input data in our model.  The output from each layer varies in dimension size, but they are all two dimensional.  The first dimension depends on the audio input length and the last dimension is constant with respect to which layer in SoundNet it came from.  The input to the MLP and SVM has to have a fixed dimensionality, so my first thought was to average the feature vector across the variable length dimension.  I used the mean and tested a 100 layer MLP with the features from layers 9-25 of SoundNet to see which had the best 5-fold cross validation accuracy. Layer 21 had the best result, so I will used that to run additional tests.  Using layer 21's features, using the mean across the variable length dimension and so the feature vector had 512 values and led to a 63.6% cross validation accuracy.  I also tested the following: maximum:61.6%, minimum:36.9%, random index:46.3%, fixed index (10): 46.2%.  I also tried taking the first 10 rows of the features (5120 size vector) and flattening them, but this only led to 53.7% accuracy.  Mean had the best results, so now I wanted to test if I could combine some of these for better results.  Using mean and concatenating with min and max improved results from 63.6% with just mean to 63.7% with all combined.  Adding standard deviation increased to 63.9% and adding the first and third quantiles again raised it to 64.8%.  So now, my feature representation of SoundNet features concatenates the mean, minimum, maxixmum, standard deviation, first quantile, and third quantile across the variable length dimension.  I tested normalizing the features to a normal distribution, however, this did not improve validation accuracy.

In order to determine which layer of extracted features using SoundNet correlate most with the labels, I trained a 100 hidden layer MLP with data from each layer individually.  Using 5-fold cross validation, I averaged the accuracy on the validation set from each fold to determine which layer yielded the best results.  The following were the layer followed by the validation accuracy: {9: 50.3%, 10: 58.3%, 11: 56.7%, 12: 54.8%, 13: 60.2%, 14: 62.5%, 15: 59.3%, 16: 62.9%, 17: 64.3%, 18: 65.1%, 19: 54.6%, 20: 63.4%, 21: 65.6%, 22: 19.7%, 23: 53.4%, 24: 56.5%, 25: 41.7%}

So the best performing single layer was layer 21 followed by 18.  Using both of these layers resulted in similar validation accuracy of 66.9%, a 1.3% improvement over just layer 21.  Adding layer 14, which performed well by itself, to this combination did not improve validation accuracy.

## MFCC

Using the MFCC 50 k-means bag of features representation with SVM, resulted in 22.6% validation accuracy in the 5 fold cross validation.  This was the same as the training error, so there was no overfitting.  This indicated that perhaps the representation was too simple.  To add more complexity to the representation, I re-ran k-means clustering with 200 means rather than 50.  This process took 1 hour and 46 minutes to complete.

Training SVM on the 200 means bog of features, led to an a validation accuracy of 10% indicating that the model was as good as a random guess.  Perhaps with 200 means, the algorithm wasn't able to converge well and the means became meaningless (no pun intended).  This result led me to want to try less than 50 means, hypothesizing that 50 means is higher than optimal.  25 means too 679 seconds to train, and led to a 32.2% accuracy.  The accuracy with 25 means was better than 200 but worse than 50. To see if even less means lead to a better validation accuracy, I tried 10 means, however, this led to an inferior 29% 5 fold cross validation accuracy.

For my mfcc.csv output, I used a 1000 hidden unit single layer MLP with 5 times data augmentation (described below) using the concatenation of the bag-of-features sizes 10, 25, and 50.

## SVM Training

Using the out of the box code for the SVM classifier using MFCC data with 50 means led to a 22.6% 5-fold cross validation average accuracy. 

## Hyperparameter search

For deciding on hyper parameters, I used 5-fold cross validation and looped through some values.  For instance, the regularization parameter in SVM used every 0.1 values from .1 to 1 and every .5 values from 1 to 5.  The best average fold validation accuracy was 55.7% with a regularization parameter value of 2.5 indicating that we want less regularization than the default of 2.5.

For MLP, I searched for alpha values (regularization parameters) from 1e-5 to 5e-2 finding the best to be 5e-4 interms of 5-fold cross validation error.

## Data Augmentation

There really isn't a lot of training data for this problem.  Luckily we can use pretrained feature extractors.  To try to mimic having more data for this problem, I implemented a simple data augmentation technique.  I would scale up the training data by a factor by repeating points, but the repeated training points would have some gaussian noise added to them.  I hypothesized that that could help make the classifier more robust and generalize better to the unseen data.

To figure out how much noise to add, i started with a factor of two and played with adding scaled normal distributed noise.  So it had a zero mean and I would set the standard deviation. The best amount was 0.1 standard deviation which made sense since the training data had a 0.31 mean and 0.9899 standard deviation so it isn't very out of distribution but definitely adds some significant noise.  I played with a few different factors for scaling up the data.

## Best

I started my search for the best model following section "SoundNet Features".  I concatenated the mean, min, max, first and third quantiles, and standard deviation of the 18th and 21st layers of the SoundNet network.  This led to a 66.9% 5-fold cross validation accuracy.  To try to improve upon this, I added the 50 mean bag of features vectors to the feature represenation.  Again, testing using a 100 hidden layer MLP. Adding the 50 mfcc bag of features actually decreased validation accuracy by 0.3%.

Then I tried to see if a bigger model would result in better performance, I tried two 100 hidden unit layers, and various amounts of 1000 hidden unit layers.  None of these models performed better in terms of validation accuracy in 5-fold cross validation.  All attempts at decreasing the hidden units from 100 failed. Even trying two layers with 50 hidden units in each resulted in poorer validation accuracy.

My best model also uses data augmentation.  The training data is tripled with the extra samples having zero-mean 0.1 standard deviation gaussian noised added to it.  This resulted in a validation accuracy of 67.1%.

To reproduce these results, run the instructions for generating the soundnet features above, then run ```python2 train_best.py```. And to test it ```python2 test_best.py models/best.model labels/test_for_student.label best.csv``` Training it takes about 50 seconds.  Running it on the test labels takes 5.1 seconds.  The kaggle results indicate a 67.5% accuracy which is surprising since that is better than the validation accuracy.

The 5-fold cross validation confusion matrix follows where the row indicates the true value and the column indicates the predicted value.

<!-- [[427  41  15   2  22  22   7  24   9  32]
 [ 33 412  32   8  23   2   6  19  16  50]
 [ 22  17 361  35  35   4  23  76  18  10]
 [  7   8  55 362   8   4  20  28   9   8]
 [ 39  11  57   2 415  21   9  40   2   5]
 [ 25   1  11   0  22 445   4   7   4   7]
 [  5  13  22   8   4   8 402  57  65  17]
 [ 21   7  87  26  27   2  45 346  28  12]
 [  7  13  14   3   2   5  68  33 247  28]
 [ 47  66  26   7  14   3  15  21  29 373]]

with the following as class indices:

dribbling_basketball,0
mowing_lawn,1
playing_guitar,2
playing_piano,3
playing_drums,4
tapping_pen,5
blowing_out_candles,6
singing,7
tickling,8
shoveling_snow,9 -->

|                             | dribbling <br>basketball | mowing <br>lawn | playing <br>guitar | playing <br>piano | playing <br>drums | tapping<br> pen | blowing <br>out <br>candles | singing | tickling | shoveling <br>snow |
|-----------------------------|--------------------------|-----------------|--------------------|-------------------|-------------------|-----------------|-----------------------------|---------|----------|--------------------|
| dribbling <br>basketball    | 427                      | 41              | 15                 | 2                 | 22                | 22              | 7                           | 24      | 9        | 32                 |
| mowing <br>lawn             | 33                       | 412             | 32                 | 8                 | 23                | 2               | 6                           | 19      | 16       | 50                 |
| playing <br>guitar          | 22                       | 17              | 361                | 35                | 35                | 4               | 23                          | 76      | 18       | 10                 |
| playing <br>piano           | 7                        | 8               | 55                 | 362               | 8                 | 4               | 20                          | 28      | 9        | 8                  |
| playing <br>drums           | 39                       | 11              | 57                 | 2                 | 415               | 21              | 9                           | 40      | 2        | 5                  |
| tapping<br> pen             | 25                       | 1               | 11                 | 0                 | 22                | 445             | 4                           | 7       | 4        | 7                  |
| blowing <br>out <br>candles | 5                        | 13              | 22                 | 8                 | 4                 | 8               | 402                         | 57      | 65       | 17                 |
| singing                     | 21                       | 7               | 87                 | 26                | 27                | 2               | 45                          | 346     | 28       | 12                 |
| tickling                    | 7                        | 13              | 14                 | 3                 | 2                 | 5               | 68                          | 33      | 247      | 28                 |
| shoveling <br>snow          | 47                       | 66              | 26                 | 7                 | 14                | 3               | 15                          | 21      | 29       | 373                |

There are a few standouts in the confusion matrix: THe model predicted videos were of singing 76 when the video was actually of playing guitar.  This also went the other way where the model predicted playing guitar 87 times on videos that were actually of singing.  Singing and guitar playing are both musical, so it makes sense that these would be confused often.  Another highly confused class was the model predicting the videos was mowing the lawn and shoveling snow.  On the other hand, the model was extremely accurate at distinguishing the difference between mowing lawn and tapping pen.  Looking at the diagonal, you can see that the model often got dribbling basketabll, and tapping pen correct, but messed up on tickling and singing often.



## Kaggle

kaggle username @pittsburghskeet

## AWS

I performed this work on my own machine, so I still have all of my AWS credits remaining.