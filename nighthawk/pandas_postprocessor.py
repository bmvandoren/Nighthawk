import numpy as np
import pandas as pd
import pickle
import os

from postprocessor import Postprocessor
import nighthawk_new as nh


# TODO: Understand better why this slows down for small input block lengths.


quiet = True


class PandasPostprocessor(Postprocessor):


    def __init__(self, input_taxa, taxa_of_interest, threshold):

        # Create dummy values to satisfy original preprocessing code.
        # The initial postprocessor API assumes that the postprocessor
        # does not need these values, but the original preprocessing
        # code uses them.
        self.test_filename = '/test.wav'
        self.clip_length_sec = 1
        self.stride_sec = 1
        
        if not quiet:
            print("loading taxonomy") 
            
        self.class_names_dict = dict(
            (rank, input_taxa[i])
            for i, rank in enumerate(nh.TAXONOMIC_RANKS))

        # load group dictionary
        # needs to link species to group
        groups_map_df = pd.read_csv(nh.GROUP_EBIRD_CODES_FILE_PATH)
        groups_dict = dict(zip(groups_map_df.ebird_code,groups_map_df.group))

        # load taxonomy
        taxonomy_df = pd.read_csv(nh.EBIRD_TAXONOMY_FILE_PATH)
        # remove parenthetical family stuff
        taxonomy_df['family'] = taxonomy_df['family'].str.replace(' \(.*\)', '',regex=True)

        # merge group list with taxonomy
        # taxonomy_df['group'] = [*map(groups_dict.get,taxonomy_df['code'])]
        taxonomy_df['group'] = taxonomy_df['code'].map(groups_dict)

        # make mapping dictionaries from taxonomy
        self.species_group_map = dict(zip(taxonomy_df.code, taxonomy_df.group))
        self.species_family_map = dict(zip(taxonomy_df.code, taxonomy_df.family))
        self.species_order_map = dict(zip(taxonomy_df.code, taxonomy_df.order))
        self.group_family_map = dict(zip(taxonomy_df.group, taxonomy_df.family))
        self.group_order_map = dict(zip(taxonomy_df.group, taxonomy_df.order))
        self.family_order_map = dict(zip(taxonomy_df.family, taxonomy_df.order))
        
        # load test config for species subset
        if taxa_of_interest is not None:
            if not quiet:
                print("using taxon subset") 
                
            def get_taxa(taxa_of_interest, rank):
                return [
                    t for t in self.class_names_dict[rank]
                    if t in taxa_of_interest]

            self.subselect_dict = dict(
                (rank, get_taxa(taxa_of_interest, rank))
                for rank in nh.TAXONOMIC_RANKS)                        
                
        else:
            if not quiet:
                print("NOTE: making unvalidated taxon predictions") 

            self.subselect_dict = self.class_names_dict         

        if nh.CALIBRATORS_FILE_PATH is not None:
            if not quiet:
                print("loading calibrators") 
            calib_file = open(nh.CALIBRATORS_FILE_PATH, 'rb')
            self.calibrators = pickle.load(calib_file)
            calib_file.close()
        else:
            self.calibrators = None

        self.threshold = threshold / 100


    def process(self, logits, start_frame_index):

        # To satisfy the original preprocessing code, represent that
        # no input samples were out of bounds. The initial postprocessor
        # API does not handle out-of-bounds input samples.
        steps = len(logits[0])
        bad_inds = []

        # create data frames from logit predictions
        pred_df_dict = predictions_to_dfs(logits, 
                        start_frame_index,
                        steps, 
                        bad_inds, 
                        self.class_names_dict, 
                        self.subselect_dict,
                        self.clip_length_sec, 
                        self.stride_sec)
        
        # convert logits to probabilities
        probs_df_dict = {key : apply_sigmoid_df(df) for key,df in pred_df_dict.items()}
        
        # apply calibration
        if self.calibrators is not None:
            if not quiet:
                print("doing calibration") 
            probs_df_dict = {key : apply_calibration(df,self.calibrators) for key,df in probs_df_dict.items()}
        else:
            if not quiet:
                print("not doing calibration")
            
        # apply thresholds to make detections
        detect_df_dict = { key: extract_detections_from_probabilities(df,self.threshold) for key,df in probs_df_dict.items() }

        # add file path 
        for key in detect_df_dict.copy():
            detect_df_dict[key].insert(2, 'path', self.test_filename)   
            detect_df_dict[key].insert(2, 'filename', os.path.basename(self.test_filename))
            
        # merge taxonomic levels
        if not quiet:
            print("merging taxonomic predictions")
                
        merged_df = combine_taxon_detections(detect_df_dict,
                                            self.family_order_map,
                                            self.group_family_map,
                                            self.species_group_map,
                                            self.species_family_map)
        
        if not quiet:
            print("done")

        return get_detection_tuples(merged_df)


def predictions_to_dfs(predictions, 
                      start_frame_index,
                      n_pred_steps, 
                      bad_inds, 
                      class_names_dict, 
                      subselect_dict,
                      clip_length_sec, 
                      hop_size_sec):
    
    output_order = pd.DataFrame(predictions[0], columns = class_names_dict['order'])
    output_order = output_order[subselect_dict['order']]
    
    output_family = pd.DataFrame(predictions[1], columns = class_names_dict['family'])
    output_family = output_family[subselect_dict['family']]

    output_group = pd.DataFrame(predictions[2], columns = class_names_dict['group'])
    output_group = output_group[subselect_dict['group']]

    output_species = pd.DataFrame(predictions[3], columns = class_names_dict['species'])
    output_species = output_species[subselect_dict['species']]

    # remove bad indices (with bad samples, out of range, above)
    step_inds = list(range(start_frame_index, start_frame_index + n_pred_steps))
    for i in bad_inds:
        step_inds.remove(i)

    start_secs = np.array(step_inds) * hop_size_sec

    output_species['start_sec'] = start_secs
    output_species['end_sec'] = output_species['start_sec'] + clip_length_sec

    output_group['start_sec'] = start_secs
    output_group['end_sec'] = output_group['start_sec'] + clip_length_sec

    output_family['start_sec'] = start_secs
    output_family['end_sec'] = output_family['start_sec'] + clip_length_sec

    output_order['start_sec'] = start_secs
    output_order['end_sec'] = output_order['start_sec'] + clip_length_sec

    # add bad samples rows and reorder        
    def add_bad_windows(output_df):
        for i in bad_inds:
            keys = output_df.columns.tolist()
            values = [np.nan] * (output_df.shape[1]-2) + [i*hop_size_sec,i*hop_size_sec+clip_length_sec]
            new_row = pd.DataFrame(dict(zip(keys,values)),index=[0])
            output_df = output_df.append(new_row,ignore_index=True)
        output_df = output_df.sort_values(by=['start_sec'])
        return(output_df)

    output_species = add_bad_windows(output_species)
    output_group = add_bad_windows(output_group)
    output_family = add_bad_windows(output_family)
    output_order = add_bad_windows(output_order)
    
    outputs = {'order':output_order,
               'family':output_family,
               'group':output_group,
               'species':output_species}
    
    return outputs

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def apply_sigmoid_df(df):

    df_classes = df.drop(['start_sec','end_sec'],axis=1)
    df_meta = df.loc[:,['start_sec','end_sec']]

    df_classes = df_classes.apply(lambda x : sigmoid(x))
    # df_meta

    res_df = pd.concat([df_classes, df_meta], axis=1)
    return(res_df)

def extract_detections_from_probabilities(df_probs,thresh_prob=0.5):
    df = df_probs

    # drop any NA rows
    df.dropna(axis = 0, how = 'any', inplace = True)

    df_classes = df.drop(['start_sec','end_sec'],axis=1)
    df_meta = df.loc[:,['start_sec','end_sec']]

    df_bool = df_classes.apply(lambda x : x>thresh_prob)

    det_ind, det_label = np.where(df_bool)
    det_class_name = df_classes.columns[det_label]

    det_prob = np.array([df_classes.iloc[row,col] for row,col in zip(det_ind,det_label)])

    det_coords_df = df_meta.iloc[det_ind,:]
    det_values_df = pd.DataFrame({ 'class' : det_class_name,
      'prob' : det_prob },index=det_ind)

    res = pd.concat([det_coords_df,det_values_df],axis=1)
    return(res)


def apply_calibration(output_df,calibrators):
    for column in output_df:
        if column not in ['start_sec','end_sec']:
            if column in calibrators:
                output_df[column] = calibrators[column].predict(output_df[column])
            # else:
            #     print("calibrator for %s not found; not calibrating this taxon" % column)
    return output_df


def combine_taxon_detections(detect_df_dict,
                            family_order_map,
                            group_family_map,
                            species_group_map,
                            species_family_map):
    
    det_spp = detect_df_dict['species'].rename(columns={'class':'species',
                                                        'prob':'prob_species'})
    det_grp = detect_df_dict['group'].rename(columns={'class':'group',
                                                'prob':'prob_group'})
    det_fam = detect_df_dict['family'].rename(columns={'class':'family',
                                                'prob':'prob_family'})
    det_ord = detect_df_dict['order'].rename(columns={'class':'order','prob':'prob_order'})

    det = det_ord.copy()

    # map families to appropriate orders
    det_fam['order'] = det_fam['family'].map(family_order_map)

    # merge to main df based on order
    det = det.merge(det_fam,how='left',on=['start_sec','end_sec','filename','path','order'])

    # map groups to appropriate families
    det_grp['family'] = det_grp['group'].map(group_family_map)

    # merge to main df based on family
    det = det.merge(det_grp,how='left',on=['start_sec','end_sec','filename','path','family'])

    # map species to appropriate groups and families (trickier...)
    det_spp['group'] = det_spp['species'].map(species_group_map)
    det_spp['family'] = det_spp['species'].map(species_family_map)

    det_spp_groupless = det_spp[pd.isna(det_spp['group'])]
    det_spp_groupless = det_spp_groupless.drop(['group'],axis=1)
    det_spp_withgroup = det_spp[pd.notna(det_spp['group'])]

    # first merge the species with no groups based on family
    det = det.merge(det_spp_groupless,how='left',on=['start_sec','end_sec','filename','path','family'])

    # now separate data with and without groups
    det_withgroup = det[pd.notna(det['group'])]
    det_withgroup = det_withgroup.drop(['species','prob_species'],axis=1)

    det_nogroup = det[pd.isna(det['group'])]


    # then merge species with groups based on both family and group
    det_withgroup = det_withgroup.merge(det_spp_withgroup,how='left',on=['start_sec','end_sec','filename','path','family','group'])

    df_merged = pd.concat([det_nogroup,det_withgroup]).sort_values(['filename','start_sec','order','family','group','species'])


    # def float_or_na(value):
        # return float(value) if (~pd.isna(value) and (value != '<NA>')) else None

    # pdb.set_trace()

    # df_merged['prob_species'] = df_merged['prob_species'].replace({'<NA>':'nan'})
    # df_merged['prob_species'].astype(float)

    df_merged['prob_species'] = pd.to_numeric(df_merged['prob_species'], errors='coerce',downcast='float')
    df_merged['prob_group'] = pd.to_numeric(df_merged['prob_group'], errors='coerce',downcast='float')
    df_merged['prob_family'] = pd.to_numeric(df_merged['prob_family'], errors='coerce',downcast='float')
    df_merged['prob_order'] = pd.to_numeric(df_merged['prob_order'], errors='coerce',downcast='float')

    # df_merged['prob_species'] = df_merged['prob_species'].map(float_or_na)
    # df_merged['prob_group'] = df_merged['prob_group'].map(float_or_na)
    # df_merged['prob_family'] = df_merged['prob_family'].map(float_or_na)

    df_merged = df_merged.drop_duplicates()

    # remove detections with no confident order 
    df_merged = df_merged[~df_merged.order.isnull()]

    def helper1(x,col_name_list):
        x = x[col_name_list].dropna()
        if len(x)>0:
            return(x[0])
        else:
            return(pd.NA)

    if df_merged.shape[0]>0:        
        # convert to single class
        df_merged['class'] = df_merged.apply(lambda x: helper1(x,['species','group','family','order']), axis=1)          
        df_merged['prob'] = df_merged.apply(lambda x: helper1(x,['prob_species','prob_group','prob_family','prob_order']), axis=1)          
    else:
        df_merged["class"] = np.nan
        df_merged["prob"] = np.nan
    # pdb.set_trace()
    
    return df_merged


def get_detection_tuples(df):

        # Select and order DataFrame columns of interest.
        df = df[[
            'start_sec', 'order', 'prob_order', 'family', 'prob_family',
            'group', 'prob_group', 'species', 'prob_species', 'class',
            'prob']]

        # Replace NaNs with empty strings.
        df = df.fillna('')

        # Rename "start_sec" column to "frame".
        df.rename({'start_sec': 'frame'}, inplace=True)

        # Create tuples.
        return tuple(tuple(row) for row in df.values)
