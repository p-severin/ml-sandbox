import numpy

def initialize_embeddings(nb_tokens, embedding_size):
    return numpy.random.randn(nb_tokens, embedding_size)

def initialize_dense(input_size, output_size):
    weights = numpy.random.randn(input_size, output_size)
    return weights

def initialize_parameters(vocab_size, emb_size):
    dense = initialize_dense(emb_size, vocab_size)
    embeddings = initialize_embeddings(vocab_size, emb_size)

    parameters = {}
    parameters['emb'] = embeddings
    parameters['W'] = dense

    return parameters

def softmax(x):
    return numpy.divide(numpy.exp(x),
                        numpy.sum(numpy.exp(x), axis=0, keepdims=True) + 0.001)

def forward_pass(input: list, word_to_id: dict, vocab_size, emb_size):
    params = initialize_parameters(vocab_size, emb_size)
    input_array = numpy.array([params['emb'][word_to_id[word]]
                               for word
                               in input])
    output = numpy.dot(input_array, params['W'])
    output = softmax(output)
    





