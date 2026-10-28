# NLP_Masked_Language_Models
## Overview

This project compares a simple manually-built masked language model with a pretrained BERT masked language model.

## Custom Corpus

I created a custom corpus based on *The Art of Racing in the Rain*. The corpus contains original text about Enzo, Denny, racing, loyalty, perseverance, and family.

## Files

- masked_prediction_comparison.py
- corpus.txt

## Test Sentences

1. The cat sat on the [MASK]
2. I went to the [MASK] to buy food
3. She deposited money in the [MASK]
4. The trophy does not fit into the suitcase because it is too [MASK]

## Manual Model Results

The manual model relied completely on patterns found within the custom corpus and was unable to generate useful predictions because the required contexts were not present.

<img width="586" height="217" alt="image" src="https://github.com/user-attachments/assets/e8fa77b7-f178-4d27-8290-e22125442ee0" /> 
<img width="634" height="167" alt="image" src="https://github.com/user-attachments/assets/4eee04c9-e197-4516-b428-b81e19cf3d38" /> 
<img width="658" height="163" alt="image" src="https://github.com/user-attachments/assets/3d5f8771-079a-49b3-8c70-78ffa9217c87" />




## BERT Results

### The cat sat on the [MASK]
- floor
- bed
- couch

### I went to the [MASK] to buy food
- store
- market
- supermarket

### She deposited money in the [MASK]
- envelope
- bank
- box

<img width="456" height="495" alt="image" src="https://github.com/user-attachments/assets/ea996cca-b469-4f77-89d7-0afdabf266f7" />



## Failure Case

### The trophy does not fit into the suitcase because it is too [MASK]

Predictions:
- small
- heavy
- large

<img width="555" height="123" alt="image" src="https://github.com/user-attachments/assets/b81d76c5-57d0-4d98-b07e-cc77621a3708" />


This demonstrates that even BERT can struggle with ambiguity and reasoning.

## Conclusion

BERT significantly outperformed the manual model because it used context from both sides of the masked word and was trained on a much larger dataset. The assignment highlighted both the strengths and limitations of masked language models.
