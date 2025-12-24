**Deep Learning–Based Image Classification and Tourism Recommendation System**

**Project Overview**

This capstone project integrates Deep Learning (Computer Vision) and Data Science (Recommendation Systems) to solve real-world problems in the tourism domain.

The project is divided into two major parts:

Part 1: Image classification using Convolutional Neural Networks (CNNs) and Transfer Learning

Part 2: Tourism data analysis and development of a collaborative filtering–based recommendation system

The objective is to demonstrate end-to-end application of machine learning techniques, from visual data understanding to personalized recommendations.

**Technologies Used**

Python

TensorFlow / Keras

OpenCV

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

**PART 1: Deep Learning – Image Classification**

**Objective**

The goal of Part 1 is to build a CNN-based image classification model using transfer learning to classify images into different categories while handling limited data availability and preventing overfitting.

**Data Understanding and Visualization**

Plotted 8–10 sample images from each class to understand:

1. Visual patterns

2. Image resolution and quality

3. Intra-class and inter-class variations

Used the OpenCV library for image loading and visualization

This step ensured familiarity with the dataset before model development.

**CNN Architecture Selection**

Selected a pre-trained CNN architecture (e.g., ResNet)

Loaded pre-trained weights trained on large datasets such as ImageNet

Chosen due to:

1. Strong performance on image classification tasks

2. Ability to extract complex spatial features

3. Suitability for transfer learning

**Transfer Learning Strategy**

Frozen all convolutional layers of the base CNN

Retained learned low-level and high-level image features

Trained only the newly added layers on the project dataset

This significantly reduced training time and minimized overfitting.

**Model Customization**

To adapt the CNN to the dataset:

1. Added fully connected (dense) layers

2. Used appropriate activation functions

3. Applied dropout regularization to reduce overfitting

These configurations act as hyperparameters and were tuned carefully.

**Model Compilation**

The model was compiled using:

1. Optimizer: Adam

2. Loss Function: Categorical Crossentropy

3. Evaluation Metric: Accuracy

**Callbacks and Training Control**

Implemented a callback to monitor validation accuracy

Training stops automatically once validation accuracy reaches a predefined threshold

Prevents unnecessary training and overfitting

**Dataset Setup**

Defined training and validation directories

Reviewed the number of samples per class

Ensured proper dataset distribution

**Model Training (Without Augmentation)**

Trained the model without data augmentation

Continuously monitored validation accuracy

Established baseline performance

**Model Training (With Augmentation)**

Applied image augmentation techniques

Retrained the model to improve generalization

Carefully selected epochs and steps based on system capacity

**Performance Visualization**

Plotted training and validation accuracy vs epochs

Used plots to detect:

Overfitting

Underfitting

Model convergence

**PART 2: Data Science – Tourism Analysis and Recommendation System**

**Objective**

The objective of Part 2 is to analyze tourism-related datasets and build a collaborative filtering–based recommendation system that suggests tourist places based on user preferences and historical ratings.

**Data Import and Preliminary Inspection**

Imported all datasets

Checked for:

1. Missing values

2. Duplicate records

3. Anomalies and inconsistencies

Cleaned and prepared data for analysis

**User-Level Analysis**

-Age Distribution

Analyzed age distribution of users providing ratings

Identified dominant user groups contributing to tourism ratings

-Tourist Origin Analysis

Identified locations from which most tourists originate

Helped understand geographic demand trends

**Tourist Spot and Location Analysis**

Identified different categories of tourist spots

Analyzed which locations are famous for specific tourism types

Determined the best city for nature enthusiasts based on availability and popularity of nature-related spots

**Data Integration**

Merged user, place, and rating datasets

Created a unified dataset linking:

-Users

-Tourist places

-Ratings

This enabled deeper insights and recommendation modeling.

**Popularity and Preference Analysis**

Using the combined dataset:

Identified the most loved tourist spots

Determined which city has the highest number of popular spots

Analyzed which categories of places users prefer the most in Indonesia

**Recommendation System Development**

Approach Used

-Implemented a Collaborative Filtering recommendation model

-Based purely on user–place interaction data

-Does not rely on explicit place attributes

User–Item Interaction Matrix

-Rows represent users

-Columns represent tourist places

-Values represent ratings

This matrix forms the foundation of the recommendation system.

**Recommendation Logic**

For a given tourist location:

1.Identify users who rated the location

2.Find users with similar rating patterns

3.Recommend other places liked by similar users

4.Exclude places already visited or rated

**Results and Insights**

Generated personalized recommendations for users

Different users received different suggestions based on preferences

Popular places were recommended only when relevant
