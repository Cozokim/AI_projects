# :dart: Natural Language Processing & Computer Vision
- NLP : Topic modeling, with the goal to identify what is discussed by the customers in bad reviews;
- CV : Image classification, with the goal to classify automaticaly the pictures published by customers (5 categories).

# :card_index_dividers: Dataset
[Yelp Open Dataset](https://www.yelp.com/dataset)

<img src=".\pictures\yelp_open_dataset.png">

# :scroll: Tasks
## :abc: Topic Modeling
- :heavy_check_mark: Preprocessing : Tokenization, Stopwords, Lemmatization, Bigrams, Bag-of-words (BOW);
- :heavy_check_mark: Use of Latent Dirichlet Allocation (LDA): find optimal number of topics, build LDA, visualize results.

<img src=".\pictures\lda_topic_modeling.png"><img src=".\pictures\topic_worldCloud.png">

## :framed_picture: Image Classification
- :x: I don't use SIFT or ORB;
- :heavy_check_mark: I use CNN (Convolutional Neural Network);
- :heavy_check_mark: Prepare image folders structure to ease the data generation (input pipeline);
- :heavy_check_mark: Use of Google Colab Pro to leverage from its GPU;
- :heavy_check_mark: Use of VGG-16 pre-trained CNN from Keras (with Tensorflow backend);
- :heavy_check_mark: Apply transfer learning methods for the classification of the images: feature extraction + fine-tuning;
- :heavy_check_mark: Evaluate model predictions (especially mistakes).

<img src=".\pictures\vgg16_structure.png">

<img src=".\pictures\feature_learning_vs_fine_tuning.png">

# :computer: Dependencies
Google Colab GPU, Pandas, Numpy, matplotlib, scikit-learn, NLTK, Spacy, Gensim, WordCloud, Keras, Tensorflow
