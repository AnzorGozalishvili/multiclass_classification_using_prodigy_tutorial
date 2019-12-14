# multiclass_classification_using_prodigy_tutorial
In this tutorial I will show an example of multiclass text classification using prodigy and spacy. 


# Preprocess Data
See data preprocessing in ``notebooks/data_preparation.ipynb``

# Explore texts of each class and create list of terms
Files in directory: ``prodigy/terms`` contain some list of terms for each class (with `.text` extensions)

# Generate spaCy style patterns using listed terms to use boostraping technique in labeling Prodigy Labeling Tool
run following module to generate `.jsonl` files for each term
```bash
python -m prodigy.helpers.phrase_to_pattern
``` 

# Add default prodigy configuration locally and upadte home directory
add given json in `prodigy/config/prodigy.json`
```json
{
    "theme": "basic",
    "custom_theme": {},
    "batch_size": 10,
    "port": 8080,
    "host": "localhost",
    "cors": true,
    "db": "sqlite",
    "db_settings": {},
    "api_keys": {},
    "validate": true,
    "auto_create": true,
    "auto_exclude_current": true,
    "instant_submit": false,
    "feed_overlap": true,
    "show_stats": false,
    "hide_meta": false,
    "show_flag": false,
    "instructions": false,
    "swipe": false,
    "split_sents_threshold": false,
    "diff_style": "words",
    "html_template": false,
    "global_css": null,
    "javascript": null,
    "writing_dir": "ltr",
    "hide_true_newline_tokens": false,
    "ner_manual_require_click": false,
    "ner_manual_label_style": "list",
    "choice_style": "single",
    "choice_auto_accept": false,
    "darken_image": 0,
    "show_bounding_box_center": false,
    "preview_bounding_boxes": false,
    "shade_bounding_boxes": false
}
```
# For each call of prodigy you will need to run
```bash
PRODIGY_HOME=prodigy/config
```

# Update db configuration to have it locally
```json
{
    "db": "sqlite",
    "db_settings": {
        "sqlite": {
            "name": "prodigy.db",
            "path": "prodigy/db"
        }
    }
}
```

# Test if db is working
```bash
python -m prodigy.helpers.database_test
```

# Create empty dataset
```bash
PRODIGY_HOME=prodigy/config prodigy dataset toxic_comment_terms "Generate terms for each class of toxic comments"
```

# Check prodigy stats
```bash
PRODIGY_HOME=prodigy/config prodigy stats
```

# Download spacy models to have embedding vectors
```bash
python -m spacy download en_core_web_lg

```

# Teach terms if you need more suggestions from Prodigy
```bash
PRODIGY_HOME=prodigy/config prodigy terms.teach identity_hate_terms en_core_web_lg --seeds prodigy/terms/identity_hate.txt

```

# Export taught terms to jsonl format file
```bash
PRODIGY_HOME=prodigy/config prodigy terms.to-patterns identity_hate_terms prodigy/terms/identity_hate.jsonl --label IDENTITY_HATE

```

# Create dataset for classification data labeling
```bash
PRODIGY_HOME=prodigy/config prodigy dataset toxic_comments "Toxic comment classification training dataset"
```

# Teach separately for each class
From reading several community discussions I concluded that binary classification interface is the best either for 
binary or multi-label classification. It's because textcat.teach interface is focusing on one class and updating 
classification model (for boostraping) with each labeled sample. Each sample for labeling is chosen by non-confidence 
score which means that model has 50/50 for each class and is confused in prediction. If we take multiple classes at once
model will be confused to find priorities between classes. Also binary labeling goes faster since you need to click 
once only.

```bash
PRODIGY_HOME=prodigy/config prodigy textcat.teach toxic_comments en_core_web_lg data/preprocessed_labeled_data.csv --loader CSV --label TOXIC --patterns prodigy/terms/toxic.jsonl 
PRODIGY_HOME=prodigy/config prodigy textcat.teach toxic_comments en_core_web_lg data/preprocessed_labeled_data.csv --loader CSV --label THREAT --patterns prodigy/terms/threat.jsonl 
PRODIGY_HOME=prodigy/config prodigy textcat.teach toxic_comments en_core_web_lg data/preprocessed_labeled_data.csv --loader CSV --label SEVERE_TOXIC --patterns prodigy/terms/severe_toxic.jsonl 
PRODIGY_HOME=prodigy/config prodigy textcat.teach toxic_comments en_core_web_lg data/preprocessed_labeled_data.csv --loader CSV --label OBSCENE --patterns prodigy/terms/obscene.jsonl 
PRODIGY_HOME=prodigy/config prodigy textcat.teach toxic_comments en_core_web_lg data/preprocessed_labeled_data.csv --loader CSV --label INSULT --patterns prodigy/terms/insult.jsonl 
PRODIGY_HOME=prodigy/config prodigy textcat.teach toxic_comments en_core_web_lg data/preprocessed_labeled_data.csv --loader CSV --label IDENTITY_HATE --patterns prodigy/terms/identity_hate.jsonl 
```

# Train Model using Batch-Train Method
```bash
PRODIGY_HOME=prodigy/config prodigy textcat.batch-train toxic_comments en_core_web_lg --output prodigy/models/toxic_comments_classifier --factor 1 --dropout 0.2 --n-iter 10 --batch-size 10 --eval-split 0.2

```

Output of Model Training
```bash
Loaded model en_core_web_lg
Using 20% of examples (28) for evaluation
Using 100% of remaining examples (112) for training
Dropout: 0.2  Batch size: 10  Iterations: 10  

#            LOSS         F-SCORE      ACCURACY  
01           0.041        0.757        0.654                                                                                                                                                                                                  
02           0.027        0.878        0.808                                                                                                                                                                                                  
03           0.028        0.829        0.731                                                                                                                                                                                                  
04           0.020        0.829        0.731                                                                                                                                                                                                  
05           0.068        0.884        0.808                                                                                                                                                                                                  
06           0.056        0.889        0.808                                                                                                                                                                                                  
07           0.020        0.905        0.846                                                                                                                                                                                                  
08           0.024        0.878        0.808                                                                                                                                                                                                  
09           0.015        0.930        0.885                                                                                                                                                                                                  
10           0.023        0.909        0.846                                                                                                                                                                                                  

accept   accept   20
accept   reject   2 
reject   reject   3 
reject   accept   1 

Correct     23
Incorrect   3

Baseline    0.13              
Precision   0.91              
Recall      0.95              
F-score     0.93              
Accuracy    0.88


Model: /Users/anz2/PycharmProjects/multiclass_classification_using_prodigy_tutorial/prodigy/models/toxic_comments_classifier
Training data: /Users/anz2/PycharmProjects/multiclass_classification_using_prodigy_tutorial/prodigy/models/toxic_comments_classifier/training.jsonl
Evaluation data: /Users/anz2/PycharmProjects/multiclass_classification_using_prodigy_tutorial/prodigy/models/toxic_comments_classifier/evaluation.jsonl
```

# Run train-curve recipe
Evaluating a trained text classification model, determine the quality of the annotations.
```bash
PRODIGY_HOME=prodigy/config prodigy textcat.train-curve toxic_comments prodigy/models/toxic_comments_classifier --dropout 0.2 --n-iter 10 --batch-size 10 --eval-split 0.2 --n-samples 10
```

Outputs:
```bash
Starting with model prodigy/models/toxic_comments_classifier
Dropout: 0.2  Batch size: 10  Iterations: 10  Samples: 10

%            ACCURACY  
10%          0.85         +0.85                                                                                                                                                                                                               
20%          0.88         +0.04                                                                                                                                                                                                               
30%          0.92         +0.04                                                                                                                                                                                                               
40%          0.92         +0.00                                                                                                                                                                                                               
50%          0.85         -0.08                                                                                                                                                                                                               
60%          0.88         +0.04                                                                                                                                                                                                               
70%          0.88         +0.00                                                                                                                                                                                                               
80%          0.81         -0.08                                                                                                                                                                                                               
90%          0.88         +0.08                                                                                                                                                                                                               
100%         0.85         -0.04
```

# Run evaluation on any class
```bash
PRODIGY_HOME=prodigy/config prodigy textcat.eval toxic_comments prodigy/models/toxic_comments_classifier data/preprocessed_labeled_data.csv --loader CSV --label IDENTITY_HATE
```
Output:
```bash
Saved 20 annotations to database SQLite
Dataset: toxic_comments
Session ID: 2019-12-14_06-24-32


accept   accept   22
accept   reject   12
reject   reject   5 
reject   accept   0 

Correct     27 
Incorrect   12

Baseline    0.15              
Precision   0.65              
Recall      1.00              
F-score     0.79              
Accuracy    0.69
```

# Convert trained model to python package
Generate a model Python package from an existing model data directory. All data files are copied over. If the path 
to a meta.json is supplied, or a meta.json is found in the input directory, this file is used. Otherwise, the data 
can be entered directly from the command line. After packaging, you can run python setup.py sdist from the newly 
created directory to turn your model into an installable archive file.

create package
```bash
mkdir prodigy/models/toxic_comments_classifier_package
python -m spacy package prodigy/models/toxic_comments_classifier prodigy/models/toxic_comments_classifier_package
```
build package
```bash
cd prodigy/models/toxic_comments_classifier_package/en_core_web_lg-2.1.0
python setup.py sdist
```
install using pip
```bash
pip install dist/en_core_web_lg-2.1.0.tar.gz
```

# Use installed package (it updated existing en_core_web_lg model and added classifier)
```python
import spacy
nlp = spacy.load('en_core_web_lg')
text = 'fuck you bitch!'
nlp(text).cats
{'IDENTITY_HATE': 0.9136267304420471, 'INSULT': 0.9990608096122742, 'OBSCENE': 0.9992883801460266, 'SEVERE_TOXIC': 0.9417557120323181, 'THREAT': 0.05265289545059204, 'TOXIC': 0.9816978573799133}
text = 'I hate Georgian Politicians!!'
nlp(text).cats
{'IDENTITY_HATE': 0.9448535442352295, 'INSULT': 0.6212679147720337, 'OBSCENE': 0.1358739733695984, 'SEVERE_TOXIC': 0.963135838508606, 'THREAT': 0.8842633962631226, 'TOXIC': 0.8259693384170532}
```

