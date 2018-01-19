## NLP
  - Typical processing procedure might look like:
    - Lower all of your text
    - Strip out misc. spacing and punctuation
    - Remove stop words
    - Stem / Lemmatize out text
    - Part-Of-Speech Tagging
    - Expand feature matrix with N-grams
    
  - Term-Frequency matrix
    - Each column of the matrix is a word and each row is a document. Each cell
      contains the count of that word in a document.
    - Doesn't work well if documents are different length, also may have issues
      with underrepresentation due to terms with high frequency
      - One solution is to normalize the term counts by the length of a document
        which would alleviate some of the problem. L2 is default in sklearn
      - We can go further and have the value associated with a document-term
        be a measure of the importance is relation to the rest of the corpus
        (TF-IDF)
  - TF-IDF

## Stemmers

## Tokenization
Every spacy document is tokenized into sentences and further into tokens
which can be accessed by iterating the document
