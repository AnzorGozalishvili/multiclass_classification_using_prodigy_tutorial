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

# Initialize database in prodigy
init default (Sqlite) database in current directory
```bash

```