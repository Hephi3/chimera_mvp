import numpy as np
import pandas as pd
from scipy import stats

from torch.utils.data import Dataset

import os
import json
import torch

import sys
sys.path.append('/gris/gris-f/homelv/phempel/masterthesis/MMFL')  # Add path to Python path
from prepocessing import preproc_cd_file
from dataset.dataset_iterator import root_iter, id_to_filename

def get_split_by_class(dataset:Dataset, df_slice, data_dir, num_classes, page, **kwargs):
    if dataset.__class__.__name__ == 'Generic_MIL_Dataset':
        split = Generic_Split(df_slice, data_dir=data_dir, num_classes=num_classes, page=page)
    elif dataset.__class__.__name__ == 'Multimodal_Generic_MIL_Dataset':
        split = Multimodal_Generic_Split(df_slice, data_dir=data_dir, num_classes=num_classes, page=page)
    elif dataset.__class__.__name__ == 'Generic_Multi_Scale_MIL_Dataset':
        split = Generic_Multi_Scale_Split(df_slice, data_dir=data_dir, num_classes=num_classes, page=page, return_coords = kwargs.get('return_coords', False), pages = kwargs.get('pages', False))
    elif dataset.__class__.__name__ == 'MM_Multi_Scale_Dataset':
        split = MM_Multi_Scale_Split(df_slice, data_dir=data_dir, num_classes=num_classes, page=page, return_coords = kwargs.get('return_coords', False), pages = kwargs.get('pages', False))

    else:
        raise NotImplementedError(
            "Class {} is not supported for split generation.".format(dataset.__class__.__name__))
    return split    

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    
    # Filter out None splits
    valid_splits = [split_datasets[i] for i in range(len(split_datasets)) if split_datasets[i] is not None]
    valid_keys = [column_keys[i] for i in range(len(split_datasets)) if split_datasets[i] is not None]
    
    splits = [split.slide_data['slide_id'] for split in valid_splits]
    
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset)
                               for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index,
                          columns=valid_keys)

    df.to_csv(filename)
    print()
    
def load_clinical_data():
    root_iterator = root_iter(clinical_only=True)
    clinical_data = {}
    for id, data in root_iterator:
        slide_id = id_to_filename(id, he=True)
        cd_path = data['cd']
        with open(cd_path, 'r') as f:
            cd_data = json.load(f)
        
        # Preprocess clinical data
        cd_preprocessed = preproc_cd_file(cd_data)
        # Remove BRS
        del cd_preprocessed['BRS']
        
        # Flatten the clinical data
        cd_features = []
        for key in cd_preprocessed:
            if isinstance(cd_preprocessed[key], list):
                cd_features.extend(cd_preprocessed[key])
            else:
                cd_features.append(cd_preprocessed[key])
        
        clinical_data[slide_id] = cd_features
    return clinical_data


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 filter_dict={},
                 ignore=[],
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max',
                 page = 0,
                 **kwargs
                 ):
        """
        Args:
                csv_file (string): Path to the csv file with annotations.
                shuffle (boolean): Whether to shuffle
                seed (int): random seed for shuffling the data
                print_info (boolean): Whether to print a summary of the dataset
                label_dict (dict): Dictionary with key, value pairs for converting str labels to int
                ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        self.page = page

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(
            slide_data, self.label_dict, ignore, self.label_col)

        # shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()
        self.args = kwargs

    def cls_ids_prep(self):
        # store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(
                self.patient_data['label'] == i)[0]

        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        # get unique patients
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist(
            )
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()  # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id': patients,
                             'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])

        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n',
              self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' %
                  (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' %
                  (i, self.slide_cls_ids[i].shape[0]))
    

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = get_split_by_class(self, df_slice, self.data_dir, self.num_classes, page = self.page, **self.args)
        else:
            split = None

        return split

    def return_splits(self, csv_path):
        all_splits = pd.read_csv(
            csv_path, dtype=self.slide_data['slide_id'].dtype)
        train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits, 'val')
        test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None





class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 data_dir,
                 **kwargs):

        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        if type(self.data_dir) == dict:
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        full_path = os.path.join(
            data_dir, f'features_CLAM_page{self.page}', '{}.pt'.format(slide_id))
        features = torch.load(full_path)
        return features, label


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, page = 0):
        self.page = page
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)
    

    

class Multimodal_Generic_MIL_Dataset(Generic_MIL_Dataset):
    def __init__(self,
                data_dir, 
                **kwargs):
    
        self.clinical_data = load_clinical_data()
        super(Multimodal_Generic_MIL_Dataset, self).__init__(data_dir=data_dir, **kwargs)

    def __getitem__(self, idx):
        # Rufe die Basis-Implementation auf
        results = super().__getitem__(idx)
        
        # Hol die slide_id für die klinischen Daten
        slide_id = self.slide_data['slide_id'][idx]
        clinical_features = self.clinical_data.get(slide_id, None)
        clinical_features = torch.tensor(clinical_features, dtype=torch.float32) if clinical_features is not None else None

        return (*results, clinical_features)
            

        

class Multimodal_Generic_Split(Multimodal_Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, page=0):
        self.clinical_data = load_clinical_data()
        self.page = page
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)    
    
    
class Generic_Multi_Scale_MIL_Dataset(Generic_MIL_Dataset):
    def __init__(self, data_dir, pages, return_coords, **kwargs):

        super(Generic_Multi_Scale_MIL_Dataset, self).__init__(**kwargs,pages = pages, return_coords=return_coords, data_dir=data_dir)
        self.pages = pages
        self.return_coords = return_coords


    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        data_dir = self.data_dir
        page_features = []
        page_coords = []
        page_tumor_labels = []
        for page in self.pages:
            full_path = os.path.join(
                data_dir, 'features_CLAM_page{}'.format(page), '{}.pt'.format(slide_id))
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Features file not found: {full_path}")
                # print(f"Warning: Features file not found: {full_path}. Skipping this page.")
                # page_features.append([])
                continue
            features = torch.load(full_path)
            page_features.append(features)
            
            
            if self.return_coords:       
                if page == 0:
                    coords_path = os.path.join(data_dir, 'coordinates_page{}'.format(page), '{}.npy'.format(slide_id))
                else:
                    coords_path = os.path.join(data_dir, 'coordinates_page{}_scaled'.format(page), '{}.npy'.format(slide_id))
                if not os.path.exists(coords_path):
                    raise FileNotFoundError(f"Features file not found: {coords_path}")
                coords = np.load(coords_path)
                coords = np.array(coords.tolist(), dtype=np.float32)
                page_coords.append(torch.tensor(coords))        
            
        if self.return_coords:
            return page_features, label, page_coords
        else:
            return page_features, label

class Generic_Multi_Scale_Split(Generic_Multi_Scale_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, page=0, return_coords=False, pages = None):
        self.page = page
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.return_coords = return_coords
        self.pages = pages

    def __len__(self):
        return len(self.slide_data)

class MM_Multi_Scale_Dataset(Generic_Multi_Scale_MIL_Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clinical_data = load_clinical_data()
    
    def get_by_slide_id(self, slide_id):
        """
        Get the features, label and coordinates for a specific slide_id.
        """
        if not slide_id.endswith('HE'):
            slide_id = f"{slide_id}_HE"
        idx = self.slide_data['slide_id'].tolist().index(slide_id)
        return self.__getitem__(idx)
    
    def __getitem__(self, idx):
        # Rufe die Basis-Implementation auf
        results = super().__getitem__(idx)
        
        # Hol die slide_id für die klinischen Daten
        slide_id = self.slide_data['slide_id'][idx]
        clinical_features = self.clinical_data.get(slide_id, None)
        clinical_features = torch.tensor(clinical_features, dtype=torch.float32) if clinical_features is not None else None
        # input(f"Returning features for slide {slide_id} of lenth {len(results[0])} and clinical features of length {len(clinical_features) if clinical_features is not None else 'None'}")
        assert len(results) >= 3 and len(results) <=5, "Expected results to contain features, label and coords and optionally tumor labels, but got {}".format(len(results))
        
        return (*results, clinical_features, slide_id)
    
class MM_Multi_Scale_Split(MM_Multi_Scale_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, page=0, return_coords=False, pages = None):
        self.clinical_data = load_clinical_data()
        self.page = page
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.return_coords = return_coords
        self.pages = pages

    def __len__(self):
        return len(self.slide_data) 