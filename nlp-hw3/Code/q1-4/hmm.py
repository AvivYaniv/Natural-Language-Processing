import os
import time
import numpy as np
from data import *
from collections import defaultdict, Counter

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT
    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts = defaultdict(lambda: 0)
    ### YOUR CODE HERE
    DEFAULT_TAG     = 'O'
    START_TOKEN     = '*'
    STOP_TOKEN      = 'STOP'
    
    START_TUPLE     =   (START_TOKEN, START_TOKEN)
    STOP_TUPLE      =   (STOP_TOKEN,  STOP_TOKEN)
    
    def update_counts(n_items, counts):
        if n_items not in counts:
            counts[n_items] = 0
        counts[n_items] += 1
    
    def get_n_list_items(l, i, n):
        items = [x[1] for x in l[i-n+1:i+1]]
        return items[0] if 1 == len(items) else tuple(items)
        
    def get_n_counts(q_counts, sent, i, n):
        n_consequtive_tags = get_n_list_items(sent, i, n)
        update_counts(n_consequtive_tags, q_counts)
    
    for sent in sents:
        total_tokens += len(sent)
        sent = [START_TUPLE, START_TUPLE] + sent + [STOP_TUPLE]
        for i in range(1, len(sent)):
            update_counts(sent[i], e_word_tag_counts)
            
            get_n_counts(q_uni_counts, sent, i, 1)            
            get_n_counts(q_bi_counts,  sent, i, 2)              
            if i >= 2:
                get_n_counts(q_tri_counts, sent, i, 3)
            
    e_tag_counts = dict(q_uni_counts)
    e_tag_counts.pop(START_TOKEN)
    e_tag_counts.pop(STOP_TOKEN)
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts

def find_hyper_parameters(test_data, 
                          total_tokens, 
                          q_tri_counts, 
                          q_bi_counts, 
                          q_uni_counts, 
                          e_word_tag_counts,
                          e_tag_counts):
    
    def lambdas_to_string(lambda1, lambda2, lambda3):
        lambda1, lambda2, lambda3 = \
                    str(lambda1), str(lambda2), str(lambda3)
        return  f'lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3}'
    
    def print_status(accuracy, lambda1, lambda2, lambda3):
        accuracy_string = f'Accuracy: {accuracy}'
        print(' '.join([accuracy_string, lambdas_to_string(lambda1, lambda2, lambda3)]))
        
    maximal_accuracy = 0
    STEP_SIZE = 0.1
    epsilon = 0.001
    hyper_lambdas = (0, 0, 0)
    for lambda1 in np.arange(epsilon, 1.0, STEP_SIZE):
        for lambda2 in np.arange(epsilon, 1.0 - lambda1, STEP_SIZE):
            accuracy = \
                hmm_eval(
                        test_data, 
                        total_tokens, 
                        q_tri_counts, 
                        q_bi_counts, 
                        q_uni_counts, 
                        e_word_tag_counts,
                        e_tag_counts, 
                        lambda1, 
                        lambda2)
            accuracy_f1 = float(accuracy[1][2])
            if accuracy_f1 > maximal_accuracy:
                lambda3 = 1 - lambda1 - lambda2
                print(f'Accuracy: {accuracy_f1} (+{str(accuracy_f1-maximal_accuracy)})')
                maximal_accuracy = accuracy_f1
                hyper_lambdas = lambda1, lambda2, lambda3
                print_status(accuracy_f1, lambda1, lambda2, lambda3)
    print('Hyper-parameters are:')
    print_status(maximal_accuracy, *hyper_lambdas)
    # Returning hyper-parameters
    return hyper_lambdas

def transition(context):
    uni_p = float(context.q_uni_counts.get(context.tag, 0)) / total_tokens
    
    bi_p = 0
    if ((context.one_tags_back in context.q_uni_counts) and (context.tag, context.one_tags_back) in context.q_bi_counts):
        bi_p = float(context.q_bi_counts[(context.tag, context.one_tags_back)]) / context.q_uni_counts[context.one_tags_back]
        
    tri_p = 0
    if (((context.tag, context.one_tags_back, context.two_tags_back) in context.q_tri_counts and (context.one_tags_back, context.two_tags_back) in context.q_bi_counts)):
        tri_p = float(context.q_tri_counts[(context.tag, context.one_tags_back, context.two_tags_back)]) / context.q_bi_counts[(context.one_tags_back, context.two_tags_back)]
        
    probability = context.lambda1 * uni_p + context.lambda2 * bi_p + context.lambda3 * tri_p
    return None if probability == 0 else np.log(probability) 

def emission(context):
    tag_apperances_counter = context.e_tag_counts[context.tag]
    word_by_tag_apperances_counter = context.e_word_tag_counts[(context.word,context.tag)]
    emmision_p = float(word_by_tag_apperances_counter) / tag_apperances_counter
    return None if emmision_p == 0 else np.log(emmision_p) 

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    class Probabilities:
        """
        To have better runtime;
        pruning is by keeping only up to 'max_size' most probable tags
        """
        def __init__(self, max_size = 42):
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
    
    def get_cached(cache, item, calculation_function, context):
        if item not in cache:
            cache[item] = calculation_function(context)
        return cache[item]
   
    tags_probabilities = Probabilities()
    tags_probabilities.set(0, ('*','O'), 0)
        
    q_probabilities = {}
    e_probabilities = {}
    
    lambda3 = 1 - lambda1 - lambda2
    all_tags = e_tag_counts.keys()
    
    def build_context():
        class Object(object):
            pass
        context = Object()        
        context.total_tokens        = total_tokens
        context.q_tri_counts        = q_tri_counts
        context.q_bi_counts         = q_bi_counts
        context.q_uni_counts        = q_uni_counts
        context.e_word_tag_counts   = e_word_tag_counts
        context.e_tag_counts        = e_tag_counts
        context.lambda1             = lambda1
        context.lambda2             = lambda2
        context.lambda3             = lambda3        
        return context
    
    context = build_context()
    
    for i in range(1, len(sent) + 1):
        for tag in all_tags:
            maximal_probability = -float("inf")
            for (two_tags_back, one_tags_back) in tags_probabilities.get_by_index(i - 1):
                context.word = sent[i - 1][0]
                
                context.two_tags_back, context.one_tags_back, context.tag = \
                    two_tags_back, one_tags_back, tag
                    
                pi = tags_probabilities.get_by_tags(i - 1, (two_tags_back, one_tags_back))                
                q  = get_cached(q_probabilities, (two_tags_back, one_tags_back, tag), transition, context)                
                e  = get_cached(e_probabilities, (context.word, tag), emission, context)                                
                
                if None in [pi, q, e]:
                    continue
                
                # Calculated cached emission & transition with logs
                # logarithmic transformation is monotone-increasing transformation therefore keeps on probabilities order
                # Thus, reducing time of calculation for current probability by addition instead of multiplication
                current_probability = pi + q + e
                
                if current_probability > maximal_probability:
                    maximal_probability = current_probability
                    tags_probabilities.set(i, (one_tags_back, tag), maximal_probability)

    maximal_probability = -float("inf")
    predicted_two_tags_back = 'O'
    predicted_one_tags_back = 'O'
    for (two_tags_back, one_tags_back) in tags_probabilities.get_by_index(len(sent)):
        context.two_tags_back, context.one_tags_back, context.tag = two_tags_back, one_tags_back, 'STOP' 
        pi = tags_probabilities.get_by_tags(len(sent), (two_tags_back, one_tags_back))
        q  = transition(context)
        probability = pi + q
        if q is not None and probability > maximal_probability:
            maximal_probability, predicted_two_tags_back, predicted_one_tags_back = \
                probability, two_tags_back, one_tags_back
    
    predicted_tags[len(sent) - 1] = predicted_one_tags_back
    predicted_tags[len(sent) - 2] = predicted_two_tags_back
    
    for i in range(len(sent) - 3, -1, -1):
        predicted_tags[i] = \
            'O' if not tags_probabilities.get_by_index(i + 2) \
                else tags_probabilities.find_max_key(i + 2)[0]
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        
        # Entity level P/R/F1: 0.83/0.84/0.84
        lambda1 = 0.201
        lambda2 = 0.30100000000000005
    
        predicted_tags = \
            hmm_viterbi(
                sent, 
                total_tokens, 
                q_tri_counts,
                q_bi_counts, 
                q_uni_counts,
                e_word_tag_counts, 
                e_tag_counts, 
                lambda1, 
                lambda2)
        
        pred_tag_seqs.append(predicted_tags) 
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("../data/train.conll")
    dev_sents = read_conll_ner_file("../data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

#     # Finding the hyper-parameters optimal lambdas
#     find_hyper_parameters(dev_sents, 
#                           total_tokens, 
#                           q_tri_counts, 
#                           q_bi_counts, 
#                           q_uni_counts,
#                           e_word_tag_counts, 
#                           e_tag_counts)

    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
