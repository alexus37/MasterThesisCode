# Master_data

This directory contains a script to transform the data into the required format.

## create_data_set.ipynb

Create Poster data set for Mitsuba

## tshirtDataSetCreator.ipynb and tshirtDataSetCreatorPointLight.ipynb

Transforms the [DeepGarment](https://cgl.ethz.ch/Downloads/Publications/Papers/2017/Dib17a/Dib17a.pdf) dataset
into Mitsuba 2 test and train data.
To get the dataset please contact [Radek Danecek](https://inf.ethz.ch/people/person-detail.MjEyMzU2.TGlzdC8zMDQsLTg3NDc3NjI0MQ==.html)
and set the DATASET_FILE_ROOT variable to the location of the dataset.
The tshirtDataSetCreatorPointLight.ipynb creates a scene with a point light and
