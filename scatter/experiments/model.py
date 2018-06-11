from keras import backend as K
from keras import constraints, regularizers
from keras.engine.topology import Layer
from keras.layers import Input, initializers
from keras.layers import merge
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model


class AttentionLayer(Layer):
    '''
    Attention layer.
    Usage:
        lstm_layer = LSTM(dim, return_sequences=True)
        attention = AttentionLayer()(lstm_layer)
        sentenceEmb = merge([lstm_layer, attention], mode=lambda x:x[1]*x[0], output_shape=lambda x:x[0])
        sentenceEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(sentenceEmb)
    '''

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return mask

    def call(self, x, mask=None):
        multData = K.exp(K.dot(x, self.kernel))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            multData = mask * multData

        output = multData / (K.sum(multData, axis=1) + K.epsilon())[:, None]
        return output

    @staticmethod
    def get_output_shape_for(input_shape):
        newShape = list(input_shape)
        newShape[-1] = 1
        return tuple(newShape)


def createHierarchicalAttentionModel(maxSeq,
                                     embWeights=None, embeddingSize=None, vocabSize=None,  # embedding
                                     recursiveClass=GRU, wordRnnSize=100,nb_classes=3,
                                     dropWordEmb=0.2, dropWordRnnOut=0.2, dropSentenceRnnOut=0.5):
    '''
    Creates a model based on the Hierarchical Attention model according to : https://arxiv.org/abs/1606.02393
    inputs:
        maxSeq : max size for sentences
        embedding
            embWeights : numpy matrix with embedding values
            embeddingSize (if embWeights is None) : embedding size
            vocabSize (if embWeights is None) : vocabulary size
        Recursive Layers
            recursiveClass : class for recursive class. Default is GRU
            wordRnnSize : RNN size for word sequence
            sentenceRnnSize :  RNN size for sentence sequence
        Dense Layers
            wordDenseSize: dense layer at exit from RNN , on sentence at word level
            sentenceHiddenSize : dense layer at exit from RNN , on document at sentence level
        Dropout

    returns : Two models. They are the same, but the second contains multiple outputs that can be use to analyse attention.
    '''

    ##
    ## Sentence level logic
    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input')
    if embWeights is None:
        emb = Embedding(vocabSize, embeddingSize, mask_zero=True)(wordsInputs)
    else:
        emb = Embedding(embWeights.shape[0], embWeights.shape[1], mask_zero=True, weights=[embWeights],
                        trainable=False)(wordsInputs)
    if dropWordEmb != 0.0:
        emb = Dropout(dropWordEmb)(emb)
    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(emb)
    if dropWordRnnOut > 0.0:
        wordRnn = Dropout(dropWordRnnOut)(wordRnn)
    attention = AttentionLayer()(wordRnn)
    sentenceEmb = merge([wordRnn, attention], mode=lambda x: x[1] * x[0], output_shape=lambda x: x[0])
    sentenceEmb = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x: (x[0], x[2]))(sentenceEmb)
    modelSentence = Model(wordsInputs, sentenceEmb)

    documentInputs = Input(shape=(None, maxSeq), dtype='int32', name='document_input')
    sentenceMasking = Masking(mask_value=0)(documentInputs)
    sentenceEmbbeding = TimeDistributed(modelSentence)(sentenceMasking)
    sentenceRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(
        sentenceEmbbeding)
    if dropSentenceRnnOut > 0.0:
        sentenceRnn = Dropout(dropSentenceRnnOut)(sentenceRnn)
    attentionSent = AttentionLayer()(sentenceRnn)
    documentEmb = merge([sentenceRnn, attentionSent], mode=lambda x: x[1] * x[0], output_shape=lambda x: x[0])
    documentEmb = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x: (x[0], x[2]), name="att2")(documentEmb)
    documentOut = Dense(nb_classes, activation="sigmoid", name="documentOut")(documentEmb)

    model = Model(input=[documentInputs], output=[documentOut])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    return model