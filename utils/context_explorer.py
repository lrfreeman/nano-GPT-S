"""This module contains a function that demonstrates how context works in the context of a language model.
This highlights how given a context window the amount of sequence data that is used is not one per block size
but rather one less than the block size as we loop through the data extending the sequence by one each time."""

from loguru import logger

def explore_how_context_works(block_size: int = 8):
    """This function demonstrates how context works in the context of a language model."""
    # Let's say we have a sentence "the quick brown fox jumps over the lazy dog"
    sentence = "the quick brown fox jumps over the lazy dog"
    logger.info(f"Given the following sentence (assuming word level tokenisation): {sentence}")
    assert block_size < len(sentence), "block_size must be less than the length of the embedding array"
    words = sentence.split()
    
    # Let's say we are building a language model and we want to predict the next word
    # based on the context of the previous words. We will use a simple loop to demonstrate
    # how the context changes as we move along the sentence.
    count = 0
    
    x = words[:block_size]

    for i in range(len(words)-1):
        context = x[:i+1]
        target = words[i+1]
        logger.info(f"when input is {context} the target (y_h) is: {target}")
        count += 1
    logger.info(f"Total number of sequences is: {count} with a block size of {block_size}")
    
    