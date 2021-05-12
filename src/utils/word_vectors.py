from torchnlp.word_to_vector import GloVe
import os

from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors


class FastText(_PretrainedWordVectors):
    """ Enriched word vectors with subword information from Facebook's AI Research (FAIR) lab.

    A approach based on the skipgram model, where each word is represented as a bag of character
    n-grams. A vector representation is associated to each character n-gram; words being
    represented as the sum of these representations.

    References:
        * https://arxiv.org/abs/1607.04606
        * https://fasttext.cc/
        * https://arxiv.org/abs/1710.04087

    Args:
        language (str): language of the vectors
        aligned (bool): if True: use multilingual embeddings where words with
            the same meaning share (approximately) the same position in the
            vector space across languages. if False: use regular FastText
            embeddings. All available languages can be found under
            https://github.com/facebookresearch/MUSE#multilingual-word-embeddings
        cache (str, optional): directory for cached vectors
        unk_init (callback, optional): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be any function that takes in a Tensor and
            returns a Tensor of the same size
        is_include (callable, optional): callable returns True if to include a token in memory
            vectors cache; some of these embedding files are gigantic so filtering it can cut
            down on the memory usage. We do not cache on disk if ``is_include`` is defined.

    Example:
        >>> from torchnlp.word_to_vector import FastText  # doctest: +SKIP
        >>> vectors = FastText()  # doctest: +SKIP
        >>> vectors['hello']  # doctest: +SKIP
        -0.1595
        -0.1826
        ...
        0.2492
        0.0654
        [torch.FloatTensor of size 300]
    """
    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'
    aligned_url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{}.align.vec'

    def __init__(self, language="en", aligned=False, **kwargs):
        if aligned:
            url = self.aligned_url_base.format(language)
        else:
            url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


def load_word_vectors(word_vectors_name, embedding_size, word_vectors_cache='../data/word_vectors_cache'):

    implemented_vector_embeddings = ('GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en')
    assert word_vectors_name in implemented_vector_embeddings

    word_vectors = None

    if word_vectors_name == 'GloVe_6B':
        assert embedding_size in (50, 100, 200, 300)
        word_vectors = GloVe(name='6B', dim=embedding_size, cache=word_vectors_cache)

    if word_vectors_name == 'GloVe_42B':
        embedding_size = 300
        word_vectors = GloVe(name='42B', cache=word_vectors_cache)

    if word_vectors_name == 'GloVe_840B':
        embedding_size = 300
        word_vectors = GloVe(name='840B', cache=word_vectors_cache)

    if word_vectors_name == 'GloVe_twitter.27B':
        assert embedding_size in (25, 50, 100, 200)
        word_vectors = GloVe(name='twitter.27B', dim=embedding_size, cache=word_vectors_cache)

    if word_vectors_name == 'FastText_en':
        embedding_size = 300
        word_vectors = FastText(language='en', cache=word_vectors_cache)

    return word_vectors, embedding_size

# vectors = FastText()  # doctest: +SKIP
# print(vectors['hello'])  # doctest: +SKIP