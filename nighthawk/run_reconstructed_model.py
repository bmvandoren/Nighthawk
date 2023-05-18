import librosa
import time
import numpy as np
import pandas as pd
import os
import json
from itertools import compress

from nighthawk.probability_calibration_utils import load_calibrations


def process_overlapping_detections(df,combine_type,
                                  intervals_closed=True,
                                  max_duration=5):

    df.sort_values("start_sec", inplace=True)

    # drop any with predicted_category = NA
    df = df[~pd.isna(df['predicted_category'])]
    
    if len(df)==0:
        return df
    
    # group detections by predicted_category
    df_list = [d for _, d in df.groupby('predicted_category', group_keys=False)]

    # make merge_group for each df in list
    def merge_detections(group):
        if len(group)==1:
            return group

        which_row_max = np.argmax(group['prob'])

        new_row = group.iloc[which_row_max]
        new_row['start_sec'] = np.min(group['start_sec'])
        new_row['end_sec'] = np.max(group['end_sec'])
        # convert to one-row data frame
        new_row = new_row.to_frame().T
        # display(new_row)
        return new_row
    
    def drop_single_detections(group):
        if len(group)==1:
            # pdb.set_trace()
            return group.reset_index(drop=True).drop(0) # return df with no rows
        else:
            return group
        
    if combine_type=='merge':
        combine_func = merge_detections
    elif combine_type=='drop':
        combine_func = drop_single_detections
    else:
        raise Exception("combine_type must be either 'merge' or 'drop'")

    # for each predicted_category, sort by start_time and group again by whether the detections are overlapping
    for i in range(len(df_list)):
        df_list[i].sort_values("start_sec", inplace=True)
        
        if intervals_closed: # abutting intervals are merged
            merge_group = (df_list[i]["start_sec"]>df_list[i]["end_sec"].shift()).cumsum()
        else: # intervals must not only touch, but actually overlap
            merge_group = (df_list[i]["start_sec"]>=df_list[i]["end_sec"].shift()).cumsum()

        # now combine
        df_list[i] = df_list[i].groupby(merge_group, group_keys=False).apply(combine_func)

    if len(df['predicted_category'].unique())>1:
        df_combine = pd.concat(df_list)
    else:
        # if not len(df_list)>0:
        #     pdb.set_trace()
        df_combine = df_list[0]

  # code for splitting long detections
    def arange_with_t2(t1,t2,step): # puts t2 at the end
        return np.append(np.arange(start=t1,stop=t2,step=step),t2)

    def split_long_detections_helper(row,max_duration=5):
        row = row.squeeze() # convert to series
        dur = row['end_sec']-row['start_sec']
        # pdb.set_trace()
        if dur>max_duration:
            b = arange_with_t2(row['start_sec'],row['end_sec'],max_duration)
            new_starts = b[:-1]
            new_stops = b[1:]

            assert len(new_starts)==len(new_stops)
            # convert to one-row data frame
            row_df = row.to_frame().T

            new_df = pd.concat([row_df]*len(new_starts), ignore_index=True)
            new_df['start_sec'] = new_starts
            new_df['end_sec'] = new_stops

            return new_df
        else:
            return row.to_frame().T


    def split_long_detections(df,max_duration=5):
        df = df.reset_index(drop=True)
        is_too_long = df['end_sec']-df['start_sec'] > max_duration
        df_keep = df.loc[~is_too_long]
        df_split = df.loc[is_too_long]

        df_split['tmp'] = range(df_split.shape[0])
        df_split = df_split.groupby('tmp', group_keys=False).apply(split_long_detections_helper,max_duration=max_duration)
        df_split = df_split.drop('tmp',axis=1)
        df_split = df_split.reset_index(drop=True)

        df_out = pd.concat([df_keep,df_split])
        df_out = df_out.sort_values('start_sec').reset_index(drop=True)
        return df_out        
        
    # split any detections longer than max_duration
    df_combine = split_long_detections(df_combine,max_duration=max_duration)
    
    # splitting already does this at the end
    # df_combine = df_combine.sort_values('start_sec').reset_index(drop=True)

    return df_combine

# take a merged_df with combined taxa, and extract a particular taxonomic level (e.g. family).
def extract_tax_from_merged(df,tax_level):
    df_tax = df[['start_sec','end_sec','filename','path',tax_level,'prob_' + tax_level]].copy()
    df_tax.rename(columns = {tax_level:'predicted_category', 'prob_' + tax_level:'prob'}, inplace = True)
    return df_tax

# remove detections *equal to* or shorter than length_threshold
def remove_detections_by_duration(detect_df,length_threshold,hop_size_sec):
    dur = detect_df['end_sec']-detect_df['start_sec']
    detect_df = detect_df[dur>(length_threshold+hop_size_sec/2)] # add an epsilon less than hop_size_sec to account for any rounding error
    return detect_df

def merge_tax_separately(df,tax_level,combine_type,
                         filter_clip_length,
                         filter_hop_size,
                         display_dfs=False):
    
    # take a merged_df with combined taxa, and extract a particular taxonomic level (e.g. family)
    df_tax = extract_tax_from_merged(df,tax_level)
    
    # if display_dfs:
    #     display(df_tax)

    # process overlapping detections using one of two possible options:
    #   (1) 'merge' - merge detections accompanied by an overlapping detection of the same predicted_category; pass forward the maximum probability value
    #   (2) 'drop' - drop any detections not accompanied by an overlapping detection of the same predicted_category
    df_tax = process_overlapping_detections(df_tax,combine_type=combine_type)

    # if display_dfs:
    #     display(df_tax)

    # if we use the 'merge' option, remove any detections *equal to* or shorter than the original length - this will get rid of any detections that did not have any overlaps (which would by definition have a longer duration after being merged with an overlapping detection)
    if combine_type=='merge':
        if filter_clip_length is not None and filter_hop_size is not None:
            df_tax = remove_detections_by_duration(df_tax,filter_clip_length,filter_hop_size)

    # if display_dfs:
    #     display(df_tax)

    return df_tax



def predictions_to_dfs(predictions, 
                      n_pred_steps, 
                      bad_inds, 
                      predicted_category_names_dict, 
                       subselect_dict,
                      clip_length_sec, 
                      hop_size_sec):
    
    output_order = pd.DataFrame(predictions[0], columns = predicted_category_names_dict['order'])
    output_order = output_order[subselect_dict['order']]
    
    output_family = pd.DataFrame(predictions[1], columns = predicted_category_names_dict['family'])
    output_family = output_family[subselect_dict['family']]

    output_group = pd.DataFrame(predictions[2], columns = predicted_category_names_dict['group'])
    output_group = output_group[subselect_dict['group']]

    output_species = pd.DataFrame(predictions[3], columns = predicted_category_names_dict['species'])
    output_species = output_species[subselect_dict['species']]

    # remove bad indices (with bad samples, out of range, above)
    step_inds = list(range(0,n_pred_steps))
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
    




# function to process chunk or file

def audio_to_examples(audio_samples,
                      sr,
                      clip_length_sample,
                      stride_sec):
                      
                      
    start_sec = 0.
    start_sample = int(start_sec * sr)
    end_sample = start_sample + clip_length_sample

    to_evaluate = []
    bad_inds = []
    step_counter = 0
    while end_sample <= len(audio_samples):

        example = audio_samples[start_sample:end_sample]

        if np.max(example)>1.1 or np.min(example)<(-1.1):
            print("max:",np.max(example),"min:",np.min(example))
            print("%s [%f,%f]" % (start_sample/sr,end_sample/sr))
            # print(np.round(example,2))
            # pdb.set_trace()
            # break
            # to_evaluate.append([])
            bad_inds.append(step_counter)
        else: 
            to_evaluate.append(example)

        start_sec += stride_sec
        start_sample = int(start_sec * sr)
        end_sample = start_sample + clip_length_sample
        step_counter+= 1 # increment counter
                      
    return to_evaluate, bad_inds, step_counter

def process_file_chunked(model,
                         test_filename,
                         clip_length_sec,
                         stride_sec,
                         target_sr=22050):

    # processing N-second chunks at a time - only load this amount into memory at once 

    sr = librosa.get_samplerate(test_filename)
    assert sr==target_sr, "Wrong sample rate"
    clip_length_sample = int(clip_length_sec * sr)

    test_file_dur_s = librosa.get_duration(path=test_filename)

    file_load_stride_sec = 900.

    start_sec = 0.
    end_sec = start_sec+file_load_stride_sec

    # start_sample = int(start_sec * sr)
    # end_sample = start_sample + clip_length_sample
    # end_sec = end_sample/sr


    file_start_time = time.time()

    if end_sec > test_file_dur_s:
        print("test_file %s contains less than minimum time" % (test_filename))
    else:
        pass

    predictions = None
    bad_inds = []
    step_counter = 0
    keep_going = True
    while keep_going:

        # print("from %s to %s" % (start_sec,end_sec))

        test_file_chunk, _ = librosa.load(test_filename, 
                                   sr=None,  # don't resample on load                    
                                   offset=start_sec,
                                   duration=file_load_stride_sec,
                                   dtype=np.float32)


        to_evaluate, bad_i, steps_i = audio_to_examples(test_file_chunk,sr,clip_length_sample,stride_sec)
        bad_inds.append(bad_i)
        step_counter+= steps_i

        # loop over to_evaluate - sequentially, not batched

        for sample in to_evaluate:
            # print(sample)
            # print(type(sample))

            preds = model(sample)
            
            if predictions is None:
                predictions = preds
            else:
                for i in range(len(preds)):
                    predictions[i] = np.concatenate((predictions[i],preds[i]))
        # break


        start_sec += file_load_stride_sec
        end_sec += file_load_stride_sec
        
        if start_sec >= test_file_dur_s:
            keep_going = False


    file_elapsed_time = time.time() - file_start_time
    print("processed %.1f s of audio in %.1f s (%.1fx)" % (test_file_dur_s,file_elapsed_time,
                                                  test_file_dur_s/file_elapsed_time))
    
    return predictions, bad_inds


# trying streaming

def process_file_stream(model,
                         test_filename,
                         clip_length_sec,
                         stride_sec,
                         target_sr=22050): # ADD MORE ARGUMENTS
    sr = librosa.get_samplerate(test_filename)
    assert sr==target_sr, "Wrong sample rate"
    clip_length_sample = int(clip_length_sec * sr)

    test_file_dur_s = librosa.get_duration(path=test_filename)

    start_sec = 0.
    start_sample = int(start_sec * sr)
    end_sample = start_sample + clip_length_sample

    if end_sample/sr > test_file_dur_s:
        print("test_file %s contains less than minimum time" % (test_filename))
    else:
        pass

    file_start_time = time.time()
    frame_length = clip_length_sec * sr

    stream = librosa.stream(test_filename,
                            block_length=1,               # 1 frame per block
                            frame_length=frame_length,    # 1 second per frame
                            hop_length=stride_sec*sr)

    predictions = None
    bad_inds = []
    step_counter = 0

    for sample in stream:

        if len(sample) < frame_length:
            # too few samples left for full frame

            break

        step_counter += 1
        if np.max(sample)>1.1 or np.min(sample)<(-1.1):
            print("max:",np.max(sample),"min:",np.min(sample))
            bad_inds.append(step_counter)            
        else:
            preds = model(sample)

            if predictions is None:
                predictions = preds
            else:
                for i in range(len(preds)):
                    predictions[i] = np.concatenate((predictions[i],preds[i]))
    
    file_elapsed_time = time.time() - file_start_time
    print("processed %.1f s of audio in %.1f s (%.1fx)" % (test_file_dur_s,file_elapsed_time,
                                                  test_file_dur_s/file_elapsed_time))
    
    return predictions, bad_inds, step_counter
    
def process_file_full(model,
                         test_filename,
                         clip_length_sec,
                         stride_sec,
                         target_sr=22050):
    # load all into memory first, then chunk

    sr = librosa.get_samplerate(test_filename)
    assert sr==target_sr, "Wrong sample rate"

    test_file, sr = librosa.load(test_filename, sr=None) # don't resample on load                    

    clip_length_sample = int(clip_length_sec * sr)

    test_file_dur_s = test_file.shape[0]/sr
    # total_processed_s += test_file_dur_s

    start_sec = 0.
    start_sample = int(start_sec * sr)
    end_sample = start_sample + clip_length_sample

    if end_sample > len(test_file):
        print("test_file %s contains less than minimum time" % (test_filename))
    else:
        pass

    to_evaluate, bad_inds, step_counter = audio_to_examples(test_file,sr,clip_length_sample,stride_sec)

    # loop over to_evaluate - sequentially, not batched
    file_start_time = time.time()

    predictions = None

    for sample in to_evaluate:
        
        preds = model(sample)

        if predictions is None:
            predictions = preds
        else:
            for i in range(len(preds)):
                predictions[i] = np.concatenate((predictions[i],preds[i]))

    file_elapsed_time = time.time() - file_start_time
    print("processed %.1f s of audio in %.1f s (%.1fx)" % (test_file_dur_s,file_elapsed_time,
                                                  test_file_dur_s/file_elapsed_time))\
    
    return predictions, bad_inds, step_counter

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
    det_values_df = pd.DataFrame({ 'predicted_category' : det_class_name,
      'prob' : det_prob },index=det_ind)

    res = pd.concat([det_coords_df,det_values_df],axis=1)
    return(res)


def apply_calibration(output_df,calibrators):
    for column in output_df:
        if column not in ['start_sec','end_sec']:
            if column in calibrators:
                output_df[column] = calibrators[column].predict(output_df[column])
            else:
                print("calibrator for %s not found; not calibrating this taxon" % column)
    return output_df


def combine_taxon_detections(detect_df_dict,
                            family_order_map,
                            group_family_map,
                            species_group_map,
                            species_family_map):
    
    det_spp = detect_df_dict['species'].rename(columns={'predicted_category':'species',
                                                        'prob':'prob_species'})
    det_grp = detect_df_dict['group'].rename(columns={'predicted_category':'group',
                                                'prob':'prob_group'})
    det_fam = detect_df_dict['family'].rename(columns={'predicted_category':'family',
                                                'prob':'prob_family'})
    det_ord = detect_df_dict['order'].rename(columns={'predicted_category':'order','prob':'prob_order'})

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

    merged_df = pd.concat([det_nogroup,det_withgroup]).sort_values(['filename','start_sec','order','family','group','species'])


    # def float_or_na(value):
        # return float(value) if (~pd.isna(value) and (value != '<NA>')) else None

    # pdb.set_trace()

    # merged_df['prob_species'] = merged_df['prob_species'].replace({'<NA>':'nan'})
    # merged_df['prob_species'].astype(float)

    merged_df['prob_species'] = pd.to_numeric(merged_df['prob_species'], errors='coerce',downcast='float')
    merged_df['prob_group'] = pd.to_numeric(merged_df['prob_group'], errors='coerce',downcast='float')
    merged_df['prob_family'] = pd.to_numeric(merged_df['prob_family'], errors='coerce',downcast='float')
    merged_df['prob_order'] = pd.to_numeric(merged_df['prob_order'], errors='coerce',downcast='float')

    # merged_df['prob_species'] = merged_df['prob_species'].map(float_or_na)
    # merged_df['prob_group'] = merged_df['prob_group'].map(float_or_na)
    # merged_df['prob_family'] = merged_df['prob_family'].map(float_or_na)

    merged_df = merged_df.drop_duplicates()

    # remove detections with no confident order 
    merged_df = merged_df[~merged_df.order.isnull()]

    def helper1(x,col_name_list):
        x = x[col_name_list].dropna()
        if len(x)>0:
            return(x[0])
        else:
            return(pd.NA)

    if merged_df.shape[0]>0:        
        # convert to single predicted_category
        merged_df['predicted_category'] = merged_df.apply(lambda x: helper1(x,['species','group','family','order']), axis=1)          
        merged_df['prob'] = merged_df.apply(lambda x: helper1(x,['prob_species','prob_group','prob_family','prob_order']), axis=1)          
    else:
        merged_df["predicted_category"] = np.nan
        merged_df["prob"] = np.nan
    # pdb.set_trace()
    
    return merged_df


def run_model_on_file(audio_model,
                      test_filename,
                      target_sr,
                        clip_length_sec,
                        stride_sec ,
                        species_list_fp,
                        group_list_fp,  
                        family_list_fp, 
                        order_list_fp,
                        taxonomy_fp,
                        group_map_fp,
                        calibrators_fp=None,
                      test_config_fp=None,
                         stream=False,
                         threshold = 0.5,
                     quiet=False,
                     model_runner=None,
                     postprocess_drop_singles_by_tax_level=True,
                     postprocess_merge_overlaps=True,
                     postprocess_retain_only_overlaps=True, # only does something if postprocess_merge_overlaps=True
                     mask_output_ap_threshold=None,
                      test_set_performance_dir=None
                     ):
    
    if not quiet:
        print("loading taxonomy") 
        
    def get_taxon_names(list_fp):
        df = pd.read_csv(list_fp, header = None, names =['predicted_category'])
        return list(df['predicted_category'])

    species = get_taxon_names(species_list_fp)
    groups = get_taxon_names(group_list_fp)
    families = get_taxon_names(family_list_fp)
    orders = get_taxon_names(order_list_fp)

    class_names_dict = {'species':species,
                        'group':groups,
                        'family':families,
                        'order':orders}

    (species_group_map, species_family_map, group_family_map,
        family_order_map) = load_taxonomy(taxonomy_fp, group_map_fp)
        
    # load test config for species subset
    if test_config_fp is not None:
        if not quiet:
            print("using taxon subset") 
            
        with open(test_config_fp) as f:
            test_configs = json.load(f)
            
        subselect_species = test_configs['subselect_taxa']['species']
        subselect_group = test_configs['subselect_taxa']['group']
        subselect_family = test_configs['subselect_taxa']['family']
        subselect_order = test_configs['subselect_taxa']['order']
        
        if mask_output_ap_threshold is not None and mask_output_ap_threshold>0:
            if not quiet:
                print("masking taxa with ap less than %s" % mask_output_ap_threshold)

            def intersection(lst1, lst2):
                lst3 = [value for value in lst1 if value in lst2]
                return lst3

            species_ap_fp = os.path.join(test_set_performance_dir,'taxon_summary_species.csv')
            species_ap_df = pd.read_csv(species_ap_fp)
            bad_species = species_ap_df.iloc[:,0][species_ap_df['ap_masked']<mask_output_ap_threshold].tolist()
            species_to_remove = intersection(subselect_species,bad_species)
            if not quiet:
                print("masking the following species:",species_to_remove)
            subselect_species = [x for x in subselect_species if x not in species_to_remove]
            
            group_ap_fp = os.path.join(test_set_performance_dir,'taxon_summary_group.csv')
            group_ap_df = pd.read_csv(group_ap_fp)
            bad_group = group_ap_df.iloc[:,0][group_ap_df['ap_masked']<mask_output_ap_threshold].tolist()
            group_to_remove = intersection(subselect_group,bad_group)
            if not quiet:
                print("masking the following group:",group_to_remove)
            subselect_group = [x for x in subselect_group if x not in group_to_remove]
            
            family_ap_fp = os.path.join(test_set_performance_dir,'taxon_summary_family.csv')
            family_ap_df = pd.read_csv(family_ap_fp)
            bad_family = family_ap_df.iloc[:,0][family_ap_df['ap_masked']<mask_output_ap_threshold].tolist()
            family_to_remove = intersection(subselect_family,bad_family)
            if not quiet:
                print("masking the following family:",family_to_remove)
            subselect_family = [x for x in subselect_family if x not in family_to_remove]
            
            order_ap_fp = os.path.join(test_set_performance_dir,'taxon_summary_order.csv')
            order_ap_df = pd.read_csv(order_ap_fp)
            bad_order = order_ap_df.iloc[:,0][order_ap_df['ap_masked']<mask_output_ap_threshold].tolist()
            order_to_remove = intersection(subselect_order,bad_order)
            if not quiet:    
                print("masking the following order:",order_to_remove)
            subselect_order = [x for x in subselect_order if x not in order_to_remove]        
                    
            
    else:
        if not quiet:
            print("NOTE: making unvalidated taxon predictions") 

        subselect_species = species
        subselect_group = groups
        subselect_family = families
        subselect_order = orders            

    which_spp = [i in subselect_species for i in species]
    which_grp = [i in subselect_group for i in groups]
    which_fam = [i in subselect_family for i in families]
    which_ord = [i in subselect_order for i in orders]

    # reorder subselect lists to be in the same order as the original lists
    subselect_species = list(compress(species, which_spp))
    subselect_group = list(compress(groups, which_grp))
    subselect_family = list(compress(families, which_fam))
    subselect_order = list(compress(orders, which_ord))

    subselect_taxa = subselect_species+subselect_group+subselect_family+subselect_order

    subselect_dict = {'species':subselect_species,
                            'group':subselect_group,
                            'family':subselect_family,
                            'order':subselect_order}
                
    if model_runner is not None:
        process_file = model_runner
        if not quiet:
            print("processing file: using custom model runner")
        
    elif not stream:
        process_file = process_file_full
        if not quiet:
            print("processing file: loading full audio file into memory") 
        
    else:
        process_file = process_file_stream
        if not quiet:
            print("processing file: streaming from file") 
        
    preds, bad_inds, steps = process_file(audio_model,
                                               test_filename,
                                                 clip_length_sec,
                                                 stride_sec,
                                                target_sr)
    
    if not quiet:
        print("predictions generated") 
        
    # create data frames from logit predictions
    pred_df_dict = predictions_to_dfs(preds, 
                      steps, 
                      bad_inds, 
                      class_names_dict, 
                      subselect_dict,
                      clip_length_sec, 
                      stride_sec)
    
    # convert logits to probabilities
    probs_df_dict = {key : apply_sigmoid_df(df) for key,df in pred_df_dict.items()}
    
    # apply calibration
    if calibrators_fp is not None:
        if not quiet:
            print("doing calibration") 
        calibrators = load_calibrations(calibrators_fp)
        probs_df_dict = {key : apply_calibration(df,calibrators) for key,df in probs_df_dict.items()}
    else:
        if not quiet:
            print("not doing calibration")
        
    # apply thresholds to make detections
    detect_df_dict = { key: extract_detections_from_probabilities(df,threshold) for key,df in probs_df_dict.items() }

    # add file path 
    for key in detect_df_dict.copy():
        detect_df_dict[key].insert(2, 'path', test_filename)   
        detect_df_dict[key].insert(2, 'filename', os.path.basename(test_filename))
        
    # merge taxonomic levels
    if not quiet:
        print("merging taxonomic predictions")
            
    # for additional confidence, we can choose do some postprocessing
    merged_df = postprocess(
        detect_df_dict, clip_length_sec, stride_sec, family_order_map,
        group_family_map, species_group_map, species_family_map, quiet,
        postprocess_drop_singles_by_tax_level, postprocess_merge_overlaps,
        postprocess_retain_only_overlaps)
    
    if not quiet:
        print("done")
        
    return merged_df


def load_taxonomy(taxonomy_fp, group_map_fp):

    # load group dictionary
    # needs to link species to group
    groups_map_df = pd.read_csv(group_map_fp)
    groups_dict = dict(zip(groups_map_df.ebird_code,groups_map_df.group))

    # load taxonomy
    taxonomy_df = pd.read_csv(taxonomy_fp)
    # remove parenthetical family stuff
    taxonomy_df['family'] = taxonomy_df['family'].str.replace(' \(.*\)', '',regex=True)

    # merge group list with taxonomy
    # taxonomy_df['group'] = [*map(groups_dict.get,taxonomy_df['code'])]
    taxonomy_df['group'] = taxonomy_df['code'].map(groups_dict)

    # make mapping dictionaries from taxonomy
    species_group_map = dict(zip(taxonomy_df.code, taxonomy_df.group))
    species_family_map = dict(zip(taxonomy_df.code, taxonomy_df.family))
    species_order_map = dict(zip(taxonomy_df.code, taxonomy_df.order))
    group_family_map = dict(zip(taxonomy_df.group, taxonomy_df.family))
    group_order_map = dict(zip(taxonomy_df.group, taxonomy_df.order))
    family_order_map = dict(zip(taxonomy_df.family, taxonomy_df.order))

    return (species_group_map, species_family_map, group_family_map,
        family_order_map)


def postprocess(
        detect_df_dict, clip_length_sec, stride_sec, family_order_map,
        group_family_map, species_group_map, species_family_map, quiet,
        postprocess_drop_singles_by_tax_level, postprocess_merge_overlaps,
        postprocess_retain_only_overlaps):

    # merge taxonomic levels - this also enforces taxonomic consistency on the detections
    merged_df = combine_taxon_detections(detect_df_dict,
                                        family_order_map,
                                        group_family_map,
                                        species_group_map,
                                        species_family_map)
    
    # postprocessing option 1
    # here, we look at each taxonomic level (species, group, etc.) in turn and 
    # drop any detections that are not overlapping with another detection of the same class
    if postprocess_drop_singles_by_tax_level:
         # take merged_df (i.e. with taxonomic consistency enforced), split out by taxonomic level, drop isolated detections, and recombine
        ct = 'drop'
        detect_order = merge_tax_separately(merged_df,'order',ct,clip_length_sec,stride_sec)
        detect_family = merge_tax_separately(merged_df,'family',ct,clip_length_sec,stride_sec)
        detect_group = merge_tax_separately(merged_df,'group',ct,clip_length_sec,stride_sec)
        detect_species = merge_tax_separately(merged_df,'species',ct,clip_length_sec,stride_sec)
        
        detect_df_dict = {'order':detect_order,
                          'family':detect_family,
                          'group':detect_group,
                          'species':detect_species}
                          
        merged_df = combine_taxon_detections(detect_df_dict,
                                        family_order_map,
                                        group_family_map,
                                        species_group_map,
                                    species_family_map)
    # the recombined merged_df should have non-duplicated taxon detections dropped, so 
    # only more confident detections remain

    
    # postprocessing option 2
    # here, we can choose to merge overlapping detections (or leave them as separate, possibly overlapping, windows)
    # by default, intervals are treated as closed, so detections that simply abut (but don't technical overlap) will still be merged
    if postprocess_merge_overlaps:
        ct = 'merge'
        merged_df = process_overlapping_detections(merged_df,ct)

        ## not needed currently
        # merge each of the taxonomic levels individually
        # detect_df_dict = { key: process_overlapping_detections(df,ct) for key,df in detect_df_dict.items() }

    # if we merged overlaps, should we remove detections that weren't merged?
    # check: does this do anything if postprocess_drop_singles_by_tax_level is already True?
    if postprocess_retain_only_overlaps:
        if postprocess_merge_overlaps:
            merged_df = remove_detections_by_duration(merged_df,clip_length_sec,stride_sec)

            ## not needed currently
            # calculate for each of the taxonomic levels individually
            # detect_df_dict = { key: remove_detections_by_duration(df,clip_length_sec,stride_sec) for key,df in detect_df_dict.items() }

        else:
            if not quiet:
                print("postprocess_merge_overlaps is False, so ignoring postprocess_retain_only_overlaps")

    return merged_df


def save_detections_to_file(detect_df, 
                            test_filename,
                            out_dir):
    
    test_file_prefix = os.path.basename(test_filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    detect_df.to_csv(os.path.join(out_dir, 
                              test_file_prefix + "_detections_all-taxa_HC.csv"), 
                 index=False, na_rep='')

def save_raven_selection_table(detect_df,
                               test_filename,
                               out_dir):  
    
    test_file_prefix = os.path.basename(test_filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    seltab = detect_df.rename(columns={"start_sec":'Begin Time (s)',
                                       "end_sec":'End Time (s)',
                                      "filename":'Begin File'})
    # insert low/high frequency columns after Time columns
    seltab.insert(loc = 2,
          column = 'Low Freq (Hz)',
          value = 0)
    seltab.insert(loc = 3,
          column = 'High Freq (Hz)',
          value = 11025)    

    # seltab['Begin File'] = seltab['path'].map(os.path.basename)

    seltab.to_csv(os.path.join(out_dir, 
                               test_file_prefix + "_selections_all-taxa_HC.txt"), 
                          index=False, na_rep='', sep ='\t')
