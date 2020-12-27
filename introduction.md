# Data description
Dataset was taken from {REFERENCE} for learning purpose. Original description of task:

## Context
As a Product Manager for a startup I needed to make certain data driven decisions which involved the car rental industry. So I scraped the web and built this dataset myself. I thought to make is public as a way to give back to the community.

Here is a Medium Post by me describing some insights from this dataset and releasing the dataset to public:
https://link.medium.com/Yduf7ceYC8

## Content
The content is acquired during the time of July 2020 for major US cities.

## Acknowledgements
The scraping scripts I used were built off of certain StackOverflow responses.

## Inspiration
Some of the answers this dataset can help unwind is:

* Which car makes and models are popular and in which cities
* What is the typical fare of car rental in various major cities
* Is there a Market gap or are some markets oversaturated
* Users can also explore if the ratings on the sites have any co-relation or do they appear suspicious as most are close to 5 ratings.

## Modeling
All models were trained with loss function MSE. Categorical features were one-hot encoded before train. 
The same train data (features) were used for every model 
