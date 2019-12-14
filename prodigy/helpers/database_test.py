import os

from prodigy.components.db import connect  # import the database connector

# add custom home path for loading project db
os.environ['PRODIGY_HOME'] = '.'

db = connect()  # uses the settings in your prodigy.json
db.add_dataset('test_dataset')  # add a dataset
assert 'test_dataset' in db  # check that the dataset was added

examples = [{'text': 'hello world', '_task_hash': 123, '_input_hash': 456}]
db.add_examples(examples, ['test_dataset'])  # add examples to the dataset
dataset = db.get_dataset('test_dataset')  # retrieve a dataset

assert len(dataset) == 1  # check that the examples were added

db.drop_dataset('test_dataset')
assert 'test_dataset' not in db

db.close()
