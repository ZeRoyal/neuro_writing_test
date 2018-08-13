# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle

corpus_int, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
seq_length, save_dir = pickle.load(open('params.p', mode='rb'))


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word with some randomness
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return np.random.choice(list(int_to_vocab.values()), 1, p=probabilities)[0]

gen_length = 1000
prime_words = 'она вышла на улицу и'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load the saved model
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)
    
    # Get tensors from loaded graph
    input_text = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    
    # Sentences generation setup
    gen_sentences = prime_words.split()
    #print(gen_sentences)
    prev_state = sess.run(initial_state, {input_text: np.array([[1 for word in gen_sentences]])})
    #print(prev_state)
    
    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        #print(dyn_input)
        dyn_seq_length = len(dyn_input[0])
        #print(dyn_seq_length)

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        print(probabilities[0][dyn_seq_length-1])

        pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
        
    # Remove tokens
    chapter_text = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        chapter_text = chapter_text.replace(' ' + token.lower(), key)
        
    print(chapter_text)
    
chapter_text = ' '.join(gen_sentences)
for key, token in token_dict.items():
    chapter_text = chapter_text.replace(' ' + token.lower(), key)
chapter_text = chapter_text.replace('\n ', '\n')
chapter_text = chapter_text.replace('( ', '(')
chapter_text = chapter_text.replace(' ”', '”')

capitalize_words = ['lannister', 'stark', 'lord', 'ser', 'tyrion', 'jon', 'john snow', 'daenerys', 'targaryen', 'cersei', 'jaime', 'arya', 'sansa', 'bran', 'rikkon', 'joffrey', 
                    'khal', 'drogo', 'gregor', 'clegane', 'kings landing', 'winterfell', 'the mountain', 'the hound', 'ramsay', 'bolton', 'melisandre', 'shae', 'tyrell',
                   'margaery', 'sandor', 'hodor', 'ygritte', 'brienne', 'tarth', 'petyr', 'baelish', 'eddard', 'greyjoy', 'theon', 'gendry', 'baratheon', 'baraTheon',
                   'varys', 'stannis', 'bronn', 'jorah', 'mormont', 'martell', 'oberyn', 'catelyn', 'robb', 'loras', 'missandei', 'tommen', 'robert', 'lady', 'donella', 'redwyne'
                   'myrcella', 'samwell', 'tarly', 'grey worm', 'podrick', 'osha', 'davos', 'seaworth', 'jared', 'jeyne poole', 'rickard', 'yoren', 'meryn', 'trant', 'king', 'queen',
                   'aemon']

for word in capitalize_words:
    chapter_text = chapter_text.replace(word, word.lower().title())
    
    

import  os
version_dir  = './generated-book-v1'
if not os.path.exists(version_dir):
    os.makedirs(version_dir)

num_chapters = len([name for name in os.listdir(version_dir) if os.path.isfile(os.path.join(version_dir, name))])
next_chapter = version_dir + '/chapter-' + str(num_chapters + 1) + '.md'
with open(next_chapter, "w") as text_file:
    text_file.write(chapter_text)