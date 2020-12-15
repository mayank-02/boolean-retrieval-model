# Boolean Retrieval Model

The class `BooleanModel.py` implements a toy search engine to illustrate the boolean retrieval model for text documents.

The program asks you to enter a search query, and then returns all documents matching the query (exact match), in no particular order (unranked retrieval).

The document corpus consists of documents, which are short stories downloaded from [here](https://www.rong-chang.com/qa2/).

## Getting Started

- Install Python 3.6+
- Install all pip requirements from the `requirements.txt`:

```bash
$ python3 -m pip install -r requirements.txt
```

- To download stopwords used for the model, open your terminal or command prompt and enter following commands:

```bash
$ python3
>>> import nltk
>>> nltk.download('stopwords')
```

## Usage

```python
# Import boolean model
from BooleanModel import BooleanModel

# Create a model on your corpus of documents by passing it's path as an argument
model = BooleanModel("./corpus/*")

# Query on it as many times as you like
results = model.query("book")

# results = ['Freeway Chase Ends at Newsstand.txt', 'A Festival of Books.txt']

# Querying on a word which is not in the corpus
results = model.query("pikachu")

# Warning: pikachu was not found in the corpus!
# results = []
```

### Queries

#### Supported Queries

- Single term => `ash`
- AND => `ash & may`
- OR => `ash | may & brown`
- Parenthesis => `( ash | may ) & brown`
- NOT => `( ~ash | may ) & brown`

> Precedence: NOT (~) > AND (&) > OR (|)

#### Unsupported Queries

- NOT operator on an  intermediate result => `~( ash | may ) & brown`
- Spaces between NOT operator and operand => `~ ash & may`

## Methodology

1. Preprocessing to build standard inverted index
   - Remove special characters
   - Remove digits
   - Tokenize
   - Lowercasing
   - Stemming using `PorterStemmer`
   - Add unique words and their postings to the index

2. Refer to [this](https://nlp.stanford.edu/IR-book/html/htmledition/an-example-information-retrieval-problem-1.html) for the internals of boolean model and query evaluation

## Note

- In case of start byte invalid errors, check for character encodings of the documents in corpus. (Currently, `utf-8` is used.)

## Authors

[Mayank Jain](https://github.com/mayank-02)

## License

MIT
