from .reuters21578 import Reuters_Dataset
from .newsgroups20 import Newsgroups20_Dataset
from .imdb import IMDB_Dataset
from .pretrain_reuters21578 import Pretrain_Reuters_Dataset
from .openwebtext import OpenWebText_Dataset

def load_dataset(dataset_name, data_path, normal_class, tokenizer='spacy', use_tfidf_weights=False,
                 append_sos=False, append_eos=False, clean_txt=False, load_encoder=False, save_encoder=False):
    """Loads the dataset."""

    implemented_datasets = ('reuters', 'newsgroups20', 'imdb', 'reuters-pretrain', 'openwebtext')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'reuters':
        dataset = Reuters_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                  use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                                  clean_txt=clean_txt, load_encoder=load_encoder)

    if dataset_name == 'newsgroups20':
        dataset = Newsgroups20_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                       use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                       append_eos=append_eos, clean_txt=clean_txt, load_encoder=load_encoder)

    if dataset_name == 'imdb':
        dataset = IMDB_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                               use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                               clean_txt=clean_txt, load_encoder=load_encoder)

    if dataset_name == 'reuters-pretrain':
        dataset = Pretrain_Reuters_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                  use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                                  clean_txt=clean_txt)

    if dataset_name == 'openwebtext':
        dataset = OpenWebText_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                                clean_txt=clean_txt, save_encoder=save_encoder)
    
    return dataset
