# FinalProjectCISC642
Final project for CISC642: Introduction to Computer Vision

To access the full dataset we used, a compressed version is available to download [here](https://drive.google.com/open?id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo)

Additionally, we have a subset of the dataset, containing 100 total items (20 per category) for both the test and train sets.

A directory of compressed results from the Geometric Matching Module (GMM) can be found [here](https://drive.google.com/open?id=1eOS3bv0h7cuB54EzpjqsWRgAEqHjJGxZ)

A directory of compressed results from the Try-On Module can be found [here](https://drive.google.com/open?id=1JN2lr0zGCy35YB9tRxbRal9ZwrImmfDu)

Additionally, we have a subset of the warped-cloth and warped-mask from the train and test set for the GMM and a subset of virtual try-on result images from the train and test set for the Try-On module.

The source code for the code used for this project is found in directory /final_project

Please move the sample dataset or the full dataset to the /data directory inside /final_project. 

To train the Geometric Matching Module (GMM):

python gmm_main.py --name gmm_train --stage train --datamode train

To perfom inference:

python gmm_main.py --name gmm_test --stage test --datamode train

To train the Try-On Module (TOM):

Before training, make sure to move the generated  warp clothes  and warp clothing mask to the /data directory

python tom_main.py --name tom_train --stage train --datamode train

To perform inference:

python tom_main.py --name tom_test --stage test --datamode test

