[![HitCount](http://hits.dwyl.io/Santosh-Gupta/NaturalLanguageRecommendations.svg)](http://hits.dwyl.io/Santosh-Gupta/NaturalLanguageRecommendations)

# Natural Language Recommendations : A research paper search engine developed entirely with embedding and transformer models. 

<p align="center">
  <img src="https://i.imgur.com/r7SZcEt.png">
</p>

### Paper Data
The papers used for this project were cleaned from Semantic Scholar's Open Corpus. 
Link to the cleaned data used: https://drive.google.com/open?id=1PcdLDJUXoVXorlCTcGlM98GllArk5Z9s


### Notebooks
#### [build_index_and_search.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/build_index_and_search.ipynb)
Description: This notebook loads the trained bert model, builds the index with 1.3 million papers on TPUs and runs a demo search.

#### [create_abstract_vectors.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/create_abstract_vectors.ipynb)
Description: 

#### [inference_model.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/inference_model.ipynb)
Description: 

#### [medical_preprocessing.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/medical_preprocessing.ipynb)
Description: This notebook was used to clean the original Open Corpus dataset to retain all papers that either had a PubMed id or were part of MedLine and had at least 1 citation. Cleaned medical data in folder linked above.

#### [model.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/model.ipynb)
Description: This notebook has the training code for BERT, which is designed to run on Google Cloud TPU v3-8.

#### [pruning_first_pass.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/pruning_first_pass.ipynb)
Description: This notebook pruned our filtered data meaning that it only kept papers in the cleaned dataset that either had a citation to or were cited by another paper in the cleaned data. Pruned data in folder linked above.

#### [text2cite_preprocessing.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/text2cite_preprocessing.ipynb)
Description: This notebook was used to clean the original Open Corpus data in order to only keep papers related to fields such as engineering, math, physics, and CS. Medical/humanities papers were filtered out. Cleaned CS data in folder linked above.

#### [tfrecords_debug.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/tfrecords_debug.ipynb)
Description: Testing the tfrecord writer class.

#### [tpu_index.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/tpu_index.ipynb)
Description: 

#### [TF2.0 Word2Vec CBOW.ipynb](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/notebooks/TF2.0%20Word2Vec%20CBOW.ipynb)
Description: The original Word2Vec model implemented in Keras.

### Python files
#### [tfrecordwriter.py](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/src/TFrecordWriter.py)
Description: This file is a TFrecord writer class that has utility functions for sharing the dataset.

#### [model.py](https://github.com/Santosh-Gupta/NaturalLanguageRecommendations/blob/master/src/model.py)
Description: This is the training code for the BERT model, which is designed to run on Google Cloud TPU v3-8.
