This project is an Inca Digital challenge
URL: https://github.com/inca-digital/challenge/issues/68


Build a Deep Learning model to predict geolocation of tweets #68

Context
The goal is to create and train a deep learning model which predicts coordinates (latitude, longitude) of individual
tweets. You are free to use any approach, but we have a few suggestions. Our current idea is to use a simple
CharacterCNN architecture, that would capture the most prominent character sequences related to location-specific
language variety and probably the most common location names. We suggest avoiding using complex linguistic features
and structures in this model, specifically, Named Entity Recognition and Linking. Please apply with a rough overview
of your model architecture. No hard MSE or EER requirements - we are after a scalable model architecture that will
allow us to increase the training dataset size later on.

Development dataset
We have 4M tweets from 3,361 locations covering the South America, written in 2021. Each .csv file is named with
the coordinates (latitude_longitude) and contains the text of the tweet (column text) and some meta-information.

Deliverable
A model which takes a tweet text as input and returns the coordinates as output; the model evaluation metrics
obtained on the development dataset, including Mean Absolute Error in kilometers. We will evaluate the model using
the test dataset that is not shared here.

Resources
Read this article for inspiration.
Request access to and download the development dataset;
Message us at challenge@inca.digital with an overview of your model architecture, the obtained evaluation metrics,
and your preferred payment method. We will contact all applicants with working models, evaluate the model on
a held-out test dataset, and schedule a 30-min interview to discuss further steps. Don't hesitate to ask us
questions by commenting in this issue or emailing us at challenge@inca.digital.
