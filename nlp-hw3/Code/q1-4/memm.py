from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    ### YOUR CODE HERE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    
    # Adding features based on (Ratnaparkhi, 1996) article:
    # "A Maximum Entropy Model for Part-Of-Speech Tagging"
    
    rare_words = [word for word, rule in rare_words_transformation_rules]
    
    if curr_word in rare_words:        
        features['contains_number']         = 'containsNumber'      == curr_word
        features['contains_uppercase']      = 'containsUppercase'   == curr_word
        features['contains_hypen']          = 'containsHypen'       == curr_word
            
        PREFIX_SUFFIX_FEAUTURES_LENGTH = 5
        
        for i in range(min(len(curr_word), PREFIX_SUFFIX_FEAUTURES_LENGTH)):
            # prefixes <5/tag pairs
            features[f'prefix_size_{i}'] = curr_word[:i+1]
               
            # suffixes <5/tag pairs
            features[f'suffix_size_{i}'] = curr_word[-i-1:]
    
    # tag bigrams
    features['tag_bigrams']             = f'{prevprev_tag},{prev_tag}'
    
    # tag unigrams
    features['tag_unigrams']            = prev_tag
    
    features['prevprev_tag']            = prevprev_tag
    
    # word/tag pairs for previous word
    features['prev_word_tag']           = f'{prev_word},{prev_tag}'
    features['prevprev_word_tag']       = f'{prevprev_word},{prevprev_tag}'
    
    # word/tag pairs for subsequent word
    features['next_word']               = f'{next_word}'
    features['subsequent_word_tag']     = f'{next_word},{prev_tag}'
      
    ### YOUR CODE HERE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    for i in range(len(sent)):
        features            = extract_features(sent, i)
        vectorized_features = vectorize_features(vec, features)
        predicted_position  = logreg.predict(vectorized_features)[0]
        predicted_tag       = index_to_tag_dict[int(predicted_position)]
        predicted_tags[i]   = predicted_tag
    ### YOUR CODE HERE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    class Probabilities:
        """
        To have better runtime;
        pruning is by keeping only up to 'max_size' most probable tags
        """
        def __init__(self, max_size = 97):
            self.max_size   = max_size
            self.dictionary = defaultdict(lambda: defaultdict(dict))
    
        def get_by_index(self, i):
            return self.dictionary[i]
    
        def get_by_tags(self, i, tags_tuple):
            return self.dictionary[i][tags_tuple]
            
        def set(self, i, tags_tuple, value):
            if len(self.dictionary[i]) < self.max_size:
                self.dictionary[i][tags_tuple] = value
            else:
                min_key = self.find_min_key(i)
                if self.dictionary[i][min_key] < value:
                    self.dictionary[i].pop(min_key)
                    self.dictionary[i][tags_tuple] = value
          
        def find_min_key(self, i):
            return min(self.dictionary[i], key=self.dictionary[i].get)
        
        def find_max_key(self, i):
            return max(self.dictionary[i], key=self.dictionary[i].get)        
    
    all_tags = [v for (k,v) in index_to_tag_dict.items() if v != '*']
    
    tags_probabilities = Probabilities()
    tags_probabilities.set(0, ('*','*'), 0)
    
    for i in range(1, len(sent) + 1):
        vectorized_features = vectorize_features(vec, extract_features(sent, i - 1))
        q_probabilities     = logreg.predict_proba(vectorized_features)
        
        for tag in all_tags:
            maximal_probability = -float("inf")
            
            q = q_probabilities[0][tag_to_idx_dict[tag]]
                
            if q <= 0:
                continue
                
            for (two_tags_back, one_tags_back) in tags_probabilities.get_by_index(i - 1):
                pi = tags_probabilities.get_by_tags(i - 1, (two_tags_back, one_tags_back))
                
                current_probability = pi + np.log(q)
                if current_probability > maximal_probability:
                    maximal_probability = current_probability
                    tags_probabilities.set(i, (one_tags_back, tag), maximal_probability)

    maximal_probability = -float("inf")
    predicted_two_tags_back = 'O'
    predicted_one_tags_back = 'O'
    for (two_tags_back, one_tags_back) in tags_probabilities.get_by_index(len(sent)):
        probability = tags_probabilities.get_by_tags(len(sent), (two_tags_back, one_tags_back))
        if probability > maximal_probability:
            maximal_probability, predicted_two_tags_back, predicted_one_tags_back = \
                probability, two_tags_back, one_tags_back

    predicted_tags[len(sent) - 1] = predicted_two_tags_back
    predicted_tags[len(sent) - 2] = predicted_one_tags_back
    
    for i in range(len(sent) - 3, -1, -1):
        predicted_tags[i] = tags_probabilities.find_max_key(i + 2)[0]
    ### YOUR CODE HERE
    return predicted_tags

def should_log(sentence_index):
    if sentence_index > 0 and sentence_index % 10 == 0:
        if sentence_index < 150 or sentence_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        
        greedy_predictions  = memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        
        greedy_pred_tag_seqs.append(greedy_predictions)
        
        vietrbi_predictions = memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        
        viterbi_pred_tag_seqs.append(vietrbi_predictions)
        
        ### YOUR CODE HERE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("../data/train.conll")
    dev_sents = read_conll_ner_file("../data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
