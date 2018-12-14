# Dog Classifier

This is a project that uses machine learning to predict the species of a dog.

## Setup the Environment

Go to this link to download the image zip and annotations zip:

http://vision.stanford.edu/aditya86/ImageNetDogs/main.html

Unzip images.tgz and annotations.tgz and place in highest level directory
in the repository.

Make a virtual environment:


```git
pip install virtualenv 
virtualenv env
chmod 777 env/bin/activate
source env/bin/activate
(env) pip install -r requirements.txt
```


## Additional Information

There is a Report.pdf that has more information on this project in terms of the models used and improvements done.

## Dataset

There are 6 species we used where each species had 200 images of which were split into training and testing data. This was due to the limitation of doing this on a local machine.
