# Inca Digital Challenge #68
### Technical Report 


#### Task description

The task was to predict coordinates of a tweet given its text and some meta-information.
The data provided contains approx. 4.5G of textual data in CSV format with geotagged tweets from South America.
The reported result is an average divergence of predicted coordinates from true ones in kilometers (computed 
using Vincenty's distance formula).

#### Model description

The model is a rather straightforward implementation of a UnicodeCNN architecture from (Izbicki et al., 2019), 
done in PyTorch from scratch. Some adjustments of the initial architecture were introduced
in order to make use of the properties of the dataset. 

The model is a character-level CNN-based classifier, trained in a multi-task regime for:

  * language prediction (66 languages supported by Twitter API, out of which 53 are present in the dataset)
  * country prediction (supporting 18 countries present are in the dataset)
  * city prediction (5000 most populated cities in South America according to Simplemaps World Cities Database)

Language and country prediction were used to achieve effects of regularization for a CNN backbone, which is hard 
to train using only an MvMF loss. We do not provide an evaluation of their performance in this challenge. However, 
their results might be useful for other downstream tasks, as well as used separately from the geolocation task using 
the same model presented here.

The city prediction task is performed by calculating weights of a mixture of von Mises-Fisher distributions (MvMF)
with mean directions set to coordinates of cities in South America, which are not modified during training. 

The final prediction of a geolocation of a tweet is done by computing a center of mass of top-3 highest ranked
cities (using computed MvMF weights as masses) in the city prediction task. Although approximate, this result 
demonstrates a good compromise between computational cost and accuracy compared to a grid search approach, which is 
computationally prohibitive.

#### Data preparation

As a classifier, the model proved highly susceptible to the class imbalance present in the dataset. To mitigate that,
at least to some extent, the dataset was under-sampled to contain at most 10k samples from any given location. No 
data cleaning was attempted. The data for the model were obtained from a preprocessed textual 
collection containing only tweet's text, language, country code, and geographical coordinates from the original
data provided by the organizers. Additionally, shuffling of samples, chunking in files of size of 10k samples and 
splitting the set of files into training, testing and evaluation datasets in proportion 10:1:1 was done. 
The resulting training dataset contained approx 3.5M entries.     

#### Training procedure

We fitted the model in a multi-task regime with a combined loss of the form of a weighted sum of losses for individual 
tasks with weights 0.1 for language and country prediction tasks and 1.0 for a MvMF loss. We used an Adam optimizer 
with a learning rate of 10e-4 and an L2 regularization implemented with a PyTorch optimizer's weight decay of 10e-5.
Additionally, gradient clipping with max norm 4.0 was done. Unicode encoding was done in parallel on CPU and 
optimization was performed on GPU. Hyperparameter fine-tuning was done manually, we report the best performing model.
We trained on an entire training dataset with batches of size 1000 for 4 epochs using an 8-core Intel i7-3770 box with
16Gb RAM, a GeForce 1050Ti with 4Gb memory and an SSD, which took approx. 18 hours to complete.  

#### Evaluation results

Evaluation on a held-out dataset containing 10k samples resulted in MAE 1883.017 km, which is in line 
with the performance of the UnicodeCNN (Small) variant reported in the aforementioned paper.  

#### References

1. Izbicki et al., Geolocating Tweets in any Language at any Location.
2. World Cities Database, https://simplemaps.com/data/world-cities