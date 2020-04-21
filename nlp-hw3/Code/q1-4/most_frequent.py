import os
from data import *
from collections import defaultdict

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    tags_counts_for_each_word = {}
    # Filling a dictionary from words and tag tags to their counters
    # Going over the words and counting their tags appearances
    for sentance in train_data:
        for word, tag in sentance:
            # If first time seeing word, adding it's tags count dictionary
            if word not in tags_counts_for_each_word:
                tags_counts_for_each_word[word] = {}
            # Fetching word tags count dictionary
            word_tags_count_dictionary = tags_counts_for_each_word[word]
            # If tag not in word's tags dictionary, initializing the counter
            if tag not in word_tags_count_dictionary:
                word_tags_count_dictionary[tag] = 0
            # Incrementing word tag counter
            word_tags_count_dictionary[tag] += 1
    
    words_maximal_tags = {}
    # Going over each word and finding it's maximal tag
    for word in tags_counts_for_each_word:
        # Fetching all word tags counts
        word_tags_count_dictionary = tags_counts_for_each_word[word]
        
        maximal_tag, maximal_tag_counter = '', 0
        # Finding word tag with maximal tag counter
        for curent_tag, current_counter in word_tags_count_dictionary.items():
            if current_counter > maximal_tag_counter:
                maximal_tag, maximal_tag_counter = curent_tag, current_counter
        
        # Setting the maximal tag for current word
        words_maximal_tags[word] = maximal_tag
        
    return words_maximal_tags
    ### END CODE HERE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        DEFAULT_TAG = 'O'
        
        pred_tags_list = []
        for word in words:
            tag = DEFAULT_TAG
            if word in pred_tags:
                tag = pred_tags[word]
            pred_tags_list.append(tag)
        pred_tag_seqs.append(tuple(pred_tags_list)) 
        ### END CODE HERE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    train_sents = read_conll_ner_file("../data/train.conll")
    dev_sents = read_conll_ner_file("../data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)
    
