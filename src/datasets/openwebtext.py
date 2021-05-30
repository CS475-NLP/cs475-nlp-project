from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from nltk.corpus import reuters
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import nltk


class OpenWebText_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False, save_encoder=None):
        super().__init__(root)

        # Load the reuters dataset
        self.train_set = openwebtext_datasets(directory=root, train=True, test=False, clean_txt=clean_txt)

        # Pre-process
        self.train_set.columns.add('index')
        # self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        # self.test_set.columns.add('weight')

        # train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):


            # comment needed!!!!!!!!!!!!!!!
            # if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
            #     train_idx_normal.append(i)
            #     row['label'] = torch.tensor(0)
            # else:
            row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set)]
        # if load_encoder:
        #     # self.encoder.load_state_dict(torch.load(load_encoder))
        #     self.encoder= torch.load(load_encoder)
        #     logger.info('Loading encoder from %s.' % load_encoder)
        
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # Encode
        for row in datasets_iterator(self.train_set):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        print("self vocab size is @@@@@@", self.encoder.vocab_size)
        # Compute tf-idf weights
        print("tfidf weights", use_tfidf_weights)
        if use_tfidf_weights:
            self.test_set = None
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        
        if save_encoder:
            torch.save(self.encoder, save_encoder)
            # torch.save(self.encoder.state_dict(), save_encoder)


def openwebtext_datasets(directory='../data', train=True, test=False, clean_txt=False):
    """
    Load the Reuters-21578 dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """
    examples = []
    # openwebtext=[]

    file_path = directory+'/wikitext2.txt'

    with open(file_path, "r") as data_file:
        data = enumerate(data_file)
        for idx , text in data:
            example_text = " ".join(text.strip().split())
            # print("example text is ", example_text)
            examples.append({
            'text': example_text,
            })
    return Dataset(examples)
    


