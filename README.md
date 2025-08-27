# <div style="text-align: center;"> My Portfolio </div>

## **1. Introduction:**
I am a software developer, based out of Bangalore, India, with more than 15 years of experience in designing and developing software features and products that meet or exceed customer expectations.


I am currently developing AI/ML solutions to help customers with their business requirements, with a key focus on Large Language Models(LLMs), agentic AI development, deep learning, natural language processing (NLP), computer vision, and linear and logistic regression.

More details regarding my work portfolio are available at my GIT repository: https://github.com/AdithyaKR001/My_portfolio

I hold a post-graduate Master of Science (M.Sc.) degree in Machine Learning and Artificial Intelligence (ML and AI) from Liverpool John Moores University (LJMU), United Kingdom and a graduate engineering degree, B.E. in Computer Science and Engineering, from Anna University, Chennai, India.

My key skills include:
* Python
* Natural Language Processing.
* Deep Learning.
* Agentic AI.
* Large Language Models(LLMs).
* Linear Regression.
* Logistic Regression.
* Full-stack development.
* Java.
* Node.js

## **2. My projects:**

### **a. Large Language Models(LLMs) & Agentic AI Projects:**

#### **i.	LS Hub:AI Medical Summary Generation using LLMs:**

* Development of an application to generate AI-based LLM medical summaries to calculate the life expectancy of patients, using large-language models, to speed up medical insurance calculation by 70% and find candidates for potential insurance arbitration by 60%.

* Programming languages used: Python and Node.js.

* Python libraries used:
    * OlmOCR
    * AWS Textract
    * OpenAI
    * Flask
    * Puppeteer
* Other technologies used:
    * DigitalOcean: cloud computing platform to host the microservices for the various functionalities of the application.
    * Large Language Models used:
       * ChatGPT 5.0 Mini.
       * Gemini 2.5 Flash.
       * Grok 3.0. 

* Datasets used:
    * Historical medical records of customers and manually generated medical summaries in PDF format.
    * Expert-calculated age-wise and gender-wise mortality multiplier information in csv format.

* Processing steps:
    * OCR: conversion of input information in PDF format into textual format (.jsonl output).
    * De-identification: removal of personally identifiable information (PII) from the OCRed data (.jsonl output).
    * Summarization: generation of medical summary using the de-identified information (with program-based chunking), pre-built input prompt and expert-created mortality multiplier information (.jsonl output).
    * Report generation: generating an output PDF report in a specific format, for presentation of the AI-based medical summary from the previous step (.pdf output).

#### **ii.	SmartResolve: AI-powered RAG Agent for Quick Incident Resolution:**

* Development of a RAG (Retrieval-Augmented Generation) AI-agent to help resolve customer incidents quickly, using the knowledge base of past resolved incidents. For a particular incident, the AI agent will retrieve the top 10 similar incidents, based on the details shared in the current open incident.

* Programming language used: Python.

* Python libraries used:
    * Langchain
    * ChatOpenAI
    * OpenAIEmbeddings
    * HanaDB
    * FastAPI
* Other technologies used:
    * Databricks: for mass data pre-processing and vectorization.
    * Streamlit: front-end chat application (integrated with ServiceNow) leveraging ChatOpenAI library for conversational and visualization capabilities.
    * A cloud foundry application with API endpoints implemented using FASTAPI, integrated with the Streamlit front-end chat application, that provides the results of the incident similarity search.

* Datasets used:
    * Historical data of all the customers incidents resolved till date, obtained via APIs, from ServiceNow.

* Processing steps:
    * Data pre-processing module:redundant feature elimination, feature engineering i.e. creation of incident metadata column and incident details column, with appropriate information labelling.
    * Data Vectorization module: Chunking and storing the vectorized (embeddings) version of the incident information in the HANA vector database.
    * Similarity Search module: Takes input incident information via chat application, looks up the vector database, and returns the top 10 relevant incidents.


### **b. Natural Language Processing:**

#### **i.	Emotion and Personality Trait Detection from social media text:**

* Development of a hybrid deep-learning classification model to address existing research gaps and build a better classifier to determine the personality trait or emotion being conveyed by input lines of text gathered from social media.

* Programming language used: Python.

* Python libraries used: NLTK, Keras, Gensim (FastText, Word2Vec), GloVe, Sklearn etc.

* Datasets used:
    * MBTI
    * TEC
    * ISEAR

* Processing steps:
    * Data pre-processing steps (text cleanup, lemmatization, word tokenization etc).
    * Data splitting (train-validation-test split).
    * Text tokenization.
    * Label encoding.
    * Class imbalance handling.
    * Word embedding (FastText, GloVe,  Word2Vec).
    * CNN layers.
    * Bi-directional LSTM (or) Bi-directional GRU layer.
    * Classification layer (softmax).
    * Model performance evaluation (precision, recall, accuracy, F1 score, AUROC).


#### **ii.	Automatic Ticket Classification for a Banking Solution:**

* Development and evaluation of classifier models, to classify customer complaints based on the products (or) services, by first using NLP techniques such as topic modelling and NMF, to identify the topics (or) classes to segregate the data based on, and then building multiple supervised ML classifiers model using the same, and finally identifying the best performing classifier.
  
* Programming language used: Python.
  
* Python libraries used: Sklearn, Pandas, Wordcloud, Pandas etc.

* Dataset used: dataset of complaints received by banking customers.
  
* Processing steps:
    * Data loading and text pre-processing.
    * Exploratory data analysis (EDA).
    * Feature extraction and topic modelling.
    * Model building (supervised), training and evaluation.
    * Model inference.

#### **iii.	Disease-Treatment Matching for HealthCare application:**

* A classifier model to determine the correct treatment for the input disease symptoms was developed, using the Conditional Random Field (CRF) technique in NLP. The output was a dictionary of the symptoms and their corresponding medical treatment.

* Programming language used: Python.

* Python libraries used: sklearn_crfsuite, spacy etc.

* Processing steps:
    * Data pre-processing. 
    * Concept identification using PoS tagging.
    * Feature extraction, model building and evaluation using CRF.
    * Disease-Treatment dictionary creation.

## **c.	Computer Vision:**

#### **i.	Skin Cancer Detection:**

* Built a deep-learning based classifier to correctly detect the type of skin cancer from an input image.
  
* Programming language: Python.
  
* Python libraries used: tensorflow, keras, numpy, augmentor etc.
  
* Processing steps:
    * Data splitting.
  
    * Handling class imbalance using the augmentor library.
  
    * Improving the training data quality through image rotation, flipping etc.
  
    * Model training and evaluation.

#### **ii.	Gesture Recognition:**

* Developed a machine learning model (deep learning model) to correctly identify the gesture being performed, from an input set of images that represent a video.
  
* Programming language used: Python.
  
* Python libraries used: keras, tensorflow, cv2 etc.

* Processing steps:
    * Data augmentation and splitting.
    * Model training and evaluation.

## **d.	Other projects:**

#### **i.	Customer churn classifier for the Telecom Industry:**

* Built a prediction model for a leading company in the telecom industry to predict churn of high value customers.
  
* Programming language used: Python.
  
* Python libraries used: sklearn, matplotlib etc.
  
* Processing steps:
    * Initial analysis, data visualization and data cleanup.
  
    * Building different models with principal component analysis (PCA): logistic regression, random forest etc.
  
    * Building different models without principal component analysis (PCA): logistic regression, random forest etc.
  
    * Models’ evaluation and selection of the best performing model.
  
    * Important churn factors’ identification.

#### **ii.	Real-estate price prediction and analysis:**

* Development of an ML model to better predict housing sale prices for the real-estate industry and to better identify the key factors affecting the same.
  
* Programming language used: Python.
  
* Python libraries used: sklearn, matplotlib etc.
  
* Model processing steps:
  
    * Data cleanup, visualization, train and test data creation.
  
    * Regression model building using Ridge and Lasso, and evaluation.

#### **iii.	Bike-sharing demand prediction and analysis:**

  * Built a predictive model, to better understand the driving factors behind the demand for shared bikes and enabling the bike sharing provider to formulate a better business strategy to meet the customer’s expectations.
  
  * Programming language used: Python.

  * Python libraries used: sklearn, matplotlib, numpy etc.
    
  * Processing steps:

    * Data pre-processing: visualization, cleanup etc.
    
    * Train-test data creation.
    
    * Linear model building, with recursive feature elimination.
    
    * Model evaluation and further refinement to obtain the best performing model.
  
    * Drawing inferences from the resulting model.




