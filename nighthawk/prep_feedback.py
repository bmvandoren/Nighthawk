"""Script that prepares model feedback from users and packages it for submission to developers."""


from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path
from datetime import datetime
import tempfile
import tarfile
import os

import librosa
import pandas as pd
import numpy as np
import yaml
import soundfile as sf

import nighthawk as nh

TARGET_SR = 22050

UPDATE_COL = 'category_update'
VOC_COL = 'voc_type_update'

EXPECTED_COLUMNS = [
    #"Selection","View",     "Channel",  
    "Begin Time (s)","End Time (s)",    "Low Freq (Hz)", "High Freq (Hz)",
    #"Begin File",      "path",     "order",   
    #"prob_order",      "family",   "prob_family",     "group",   
    #"prob_group",      "species",  "prob_species",    "predicted_category",   
    #"prob",
    UPDATE_COL # special one
  ]
# ALLOWED_COLUMNS = EXPECTED_COLUMNS + [VOC_COL, "comment"]

CONFIRM_CODES = ['c','y',"C","Y"]
BACKGROUND_CODES = ['n','N','bg',"BG",'background']
UNKNOWN_CODES = ['unknown']

VOC_CODES = ['fc','fc-song','fc-long','call','song','other']

def main():

    print("\nNOTE: Please ensure that the recording start time entered\nin your YAML file is in Universal Coordinated Time (UTC).")
    
    args = _parse_args()

    # make sure arguments are valid (e.g. files exist)
    args = _check_paths(args)

    # check audio and selection table
    txt_df = _check_audio_and_txt(args)

    # print(args)  

    # get metadata (from YAML if provided); prompt for the rest
    out_filename = get_output_filename(args)

    print("\nChecks passed.")
    # copy and rename files and make an archive
    save_archive(args,txt_df,out_filename,gz=True)   
    

def _parse_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        '--audio',
        help=(f'path to the .wav file on which Nighthawk was run'),
        type=Path,
        required=True,
        dest='audio_path')
        # type=_parse_hop_size,
        # default=nh.DEFAULT_HOP_SIZE)    

    parser.add_argument(
        '--txt',
        help=(f'path to the .txt file saved by Raven with feedback'),
        type=Path,
        required=True,
        dest='txt_path')    

    parser.add_argument(
        '--yaml',
        help=(f'path to optional YAML file with recording metadata'),
        type=Path,
        required=True,
        dest='yaml_path')    
    
    parser.add_argument(
        '--output-dir',
        help=(
            'directory in which to write output package. (default: audio '
            'file directory)'),
        type=Path,
        dest='output_dir_path',
        default=None)
    
    return parser.parse_args()

def _check_paths(args):

    # make sure paths are valid
    assert args.audio_path.exists(), "provided audio path doesn't exist"
    assert args.audio_path.is_file(), "provided audio path isn't a file"
    assert args.txt_path.exists(), "provided txt path doesn't exist"
    assert args.txt_path.is_file(), "provided txt path isn't a file"

    # if args.yaml_path is not None:
    assert args.yaml_path.exists(), "provided yaml path doesn't exist"
    assert args.yaml_path.is_file(), "provided yaml path isn't a file"

    # make sure proper file types are given
    assert args.audio_path.suffix==".wav", "audio file must end in .wav"
    assert args.txt_path.suffix==".txt", "txt file must end in .txt"
    assert args.yaml_path.suffix==".yml", "yaml file must end in .yml"        

    if args.output_dir_path is not None:
        assert args.output_dir_path.exists(), "provided output directory path doesn't exist"
        assert args.output_dir_path.is_dir(), "provided output directory path is not a directory"
    else:
        args.output_dir_path = args.audio_path.parent
    
    return args
    

    # check output dir too?


def _load_taxonomy(taxonomy_fp,group_map_fp,ibp_fp):
    groups_map_df = pd.read_csv(group_map_fp)
    groups_dict = dict(zip(groups_map_df.ebird_code,groups_map_df.group))

    # load taxonomy
    taxonomy_df = pd.read_csv(taxonomy_fp)
    # remove parenthetical family stuff
    taxonomy_df['family'] = taxonomy_df['family'].str.replace(' \(.*\)', '',regex=True)

    # merge group list with taxonomy
    # taxonomy_df['group'] = [*map(groups_dict.get,taxonomy_df['code'])]
    taxonomy_df['group'] = taxonomy_df['code'].map(groups_dict)

    # load 4-letter codes
    ibp_df = pd.read_csv(ibp_fp)
    # merge 4-letter codes with taxonomy
    ibp_dict = dict(zip(ibp_df.SCINAME,ibp_df.SPEC))
    taxonomy_df['ibp_code'] = taxonomy_df['sci_name'].map(ibp_dict)
    
    return taxonomy_df


def _format_audacity_as_raven(txt_df):

    txt_df['Low Freq (Hz)'] = pd.NA
    txt_df['High Freq (Hz)'] = pd.NA

    # extract out frequency rows if present
    is_freq = txt_df.iloc[:,0]=='\\'
    if any(is_freq):
        # get corresponding time rows
        corresponding_rows = np.where(is_freq)[0]-1
        # make df with just frequency rows
        freq_df = txt_df[is_freq].iloc[:,[1,2]].copy()
        
        pd.set_option('mode.chained_assignment',None)
        txt_df['Low Freq (Hz)'][corresponding_rows] = freq_df.iloc[:,0]
        txt_df['High Freq (Hz)'][corresponding_rows] = freq_df.iloc[:,1]
        txt_df = txt_df[~is_freq]

        # convert necessary columns to numeric; raise error if there are strings present
        txt_df['Begin Time (s)'] = pd.to_numeric(txt_df['Begin Time (s)'], errors='raise')
        txt_df['End Time (s)'] = pd.to_numeric(txt_df['End Time (s)'], errors='raise')
        txt_df['Low Freq (Hz)'] = pd.to_numeric(txt_df['Low Freq (Hz)'], errors='raise')
        txt_df['High Freq (Hz)'] = pd.to_numeric(txt_df['High Freq (Hz)'], errors='raise')

    return(txt_df)    

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
        
def _check_audio_and_txt(args):

    # get duration of audio file
    audio_duration = librosa.get_duration(path=args.audio_path)

    # read txt file
    txt_df = pd.read_csv(args.txt_path, sep='\t')

    if is_float(txt_df.columns[0]) & is_float(txt_df.columns[1]):
        print("\nAudacity txt file detected.")
    
        columns_header = ['Begin Time (s)','End Time (s)','category_update']
        txt_df = pd.read_csv(args.txt_path, sep='\t', header=None,names=columns_header)
        
        txt_df = _format_audacity_as_raven(txt_df) 
    else:
        print("\nRaven txt file detected.")

    # make sure all expected columns are present
    txt_cols = txt_df.columns.tolist()
    if not all(i in txt_cols for i in EXPECTED_COLUMNS):
        missing_cols = [i for i in EXPECTED_COLUMNS if i not in txt_cols]
        assert False, "Missing column(s) in txt file: " + ', '.join(missing_cols)

    # make sure any additional columns are only the allowed ones
    # if not all(i in ALLOWED_COLUMNS for i in txt_cols):
    #     forbidden_cols = [i for i in txt_cols if i not in ALLOWED_COLUMNS]
    #     assert False, "Unrecognized column(s) in txt file: " + ', '.join(forbidden_cols)

    # make sure entries in UPDATE_COL are recognized 
        
    # get allowed taxa
    config_paths = nh.detector._get_configuration_file_paths()

    # species = pd.read_csv(config_paths.species,header=None).iloc[:,0].tolist()
    # groups = pd.read_csv(config_paths.groups,header=None).iloc[:,0].tolist()
    # families = pd.read_csv(config_paths.families,header=None).iloc[:,0].tolist()
    # orders = pd.read_csv(config_paths.orders,header=None).iloc[:,0].tolist()    

    taxonomy_df = _load_taxonomy(config_paths.ebird_taxonomy,config_paths.group_ebird_codes,config_paths.ibp_codes)

    species = taxonomy_df['code'].dropna().unique().tolist()
    groups = taxonomy_df['group'].dropna().unique().tolist()
    families = taxonomy_df['family'].dropna().unique().tolist()
    orders = taxonomy_df['order'].dropna().unique().tolist()
    ibp_codes = taxonomy_df['ibp_code'].dropna().unique().tolist()
    
    taxa = species + groups + families + orders + ibp_codes
    allowed_update_entries = taxa + BACKGROUND_CODES + UNKNOWN_CODES
    # only allow confirm codes if 'predicted_category' also present
    if 'predicted_category' in txt_cols:
        allowed_update_entries = allowed_update_entries + CONFIRM_CODES

    update_entries_lower_uniq = txt_df[UPDATE_COL].dropna()
    if len(update_entries_lower_uniq)>0:
        update_entries_lower_uniq = update_entries_lower_uniq.str.lower().str.strip().unique().tolist() # we drop NAs
        # remove any empty strings
        update_entries_lower_uniq = list(filter(None, update_entries_lower_uniq)) 
    else:
        update_entries_lower_uniq = []

    allowed_update_entries_lower = [i.lower() for i in allowed_update_entries]

    are_entries_allowed = [i in allowed_update_entries_lower for i in update_entries_lower_uniq]

    if not all(are_entries_allowed):
        bad_entries = [i for i in update_entries_lower_uniq if i not in allowed_update_entries_lower]
        assert False, "Unrecognized entries in " + UPDATE_COL +" column: " + ', '.join(bad_entries)

    # make sure entires in VOC_COL are recognized, if column is present
    if VOC_COL in txt_df.columns:
        voc_entries_lower_uniq = txt_df[VOC_COL].dropna()
        if len(voc_entries_lower_uniq)>0:
            voc_entries_lower_uniq = voc_entries_lower_uniq.str.lower().str.strip().unique().tolist() # we drop NAs
        else: 
            voc_entries_lower_uniq = []
        allowed_voc_entries_lower = [i.lower() for i in VOC_CODES]
        
        are_entries_allowed = [i in allowed_voc_entries_lower for i in voc_entries_lower_uniq]
        
        if not all(are_entries_allowed):
            bad_entries = [i for i in voc_entries_lower_uniq if i not in allowed_voc_entries_lower]
            assert False, "Unrecognized entries in " + VOC_COL +" column: " + ', '.join(bad_entries)

    # make sure no selections are longer than file duration
    assert txt_df['End Time (s)'].max() <= audio_duration, "there are selections beyond the length of the audio file"

    return txt_df
     

def get_output_filename(args):

    with open(args.yaml_path, 'r') as file:
       metadata = yaml.safe_load(file)

    _check_yml(metadata)

    recordist_name_allcaps = metadata['recordist']['name'].replace(" ", "").upper()
    location_name_allcaps = metadata['location']['name'].replace(" ", "").upper()
    lat = metadata['location']['latitude']
    lon = metadata['location']['longitude']

    assert isinstance(metadata['recording_session']['start_time_utc'], datetime), "start_time_utc is not in the valid format of YYYY-MM-DD hh:mm:ss"
    datetime_string = metadata['recording_session']['start_time_utc'].strftime("%Y%m%d_%H%M%S") + "_Z"

    lat_str = f'{lat:.2f}'
    lon_str = f'{lon:.2f}'

    name_list = [recordist_name_allcaps,
                 location_name_allcaps,
                 lat_str,
                 lon_str]
    
    # replace any spaces and underscores with hyphens
    name_list = [s.replace(' ','-').replace('_','-') for s in name_list]

    name_list = name_list + [datetime_string]
        
    output_basename = '_'.join(name_list)

    return output_basename


def save_archive(args,txt_df,out_fn,gz=True):
    
    if gz:
        mode = 'w:gz'
        ext = '.tar.gz'
    else:
        mode = 'w'
        ext = '.tar'

    archive_out_fp = args.output_dir_path.joinpath(out_fn + ext)

    # use output dir for temporary files
    # temp_dir = args.output_dir_path

    with tarfile.open(archive_out_fp, mode) as tar:
        print(f'Writing archive {archive_out_fp}.\n')

        # get sample rate of audio file
        audio_sr = librosa.get_samplerate(path=args.audio_path)

        if audio_sr != TARGET_SR:
            print(f'Input audio has sample rate {audio_sr} Hz. Resampling to {TARGET_SR} Hz.\n')
    
            y, sr_orig = librosa.load(args.audio_path,sr=None,mono=True) 
    
            y_resamp = librosa.resample(y=y, 
                                        orig_sr=sr_orig, 
                                        target_sr=TARGET_SR, 
                                        res_type='soxr_hq') # soxr_hq determined best by Harold
    
            # create a temporary file and write it to the tar
            # tmp_fp = tempfile.NamedTemporaryFile(suffix=".wav")
            tmp_wav_fp = tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
            # print("tmpfile: ",tmp_wav_fp.name)
            # print("tmpfile exists before writing?:",os.path.exists(tmp_wav_fp.name))
            sf.write(tmp_wav_fp.name, y_resamp, TARGET_SR, 'PCM_16',closefd=False)
            tar.add(tmp_wav_fp.name,arcname=out_fn + '.wav')
            # print("tmpfile exists after writing?:",os.path.exists(tmp_wav_fp.name))
            tmp_wav_fp.close()
            # print("tmpfile exists after closing?:",os.path.exists(tmp_wav_fp.name))
            os.remove(tmp_wav_fp.name)
            # print("tmpfile exists after removing?:",os.path.exists(tmp_wav_fp.name))
                        
        else:
            tar.add(args.audio_path,arcname=out_fn + '.wav')
        # tar.add(args.txt_path,arcname=out_fn + '.txt')

        # create a temporary txt file and write it to tar
        tmp_txt_fp = tempfile.NamedTemporaryFile(suffix=".txt",delete=False)
        # print("tmpfile: ",tmp_txt_fp.name)
        # print("tmpfile exists before writing?:",os.path.exists(tmp_txt_fp.name))
        txt_df.to_csv(tmp_txt_fp.name, sep='\t', index=False,mode="w+")
        tar.add(tmp_txt_fp.name,arcname=out_fn + '.txt')
        # print("tmpfile exists after writing?:",os.path.exists(tmp_txt_fp.name))
        tmp_txt_fp.close()
        # print("tmpfile exists after closing?:",os.path.exists(tmp_txt_fp.name))
        os.remove(tmp_txt_fp.name)
        # print("tmpfile exists after removing?:",os.path.exists(tmp_txt_fp.name))
        
        tar.add(args.yaml_path,arcname=out_fn + '.yml')    
    

    print(f'Done. Please send this file to Nighthawk developers.\n')

def _check_yml(metadata):
    
    if 'recordist' not in metadata:
        assert False, "'recordist' fields missing from yaml file"
    else:
        if 'name' not in metadata['recordist']:
            assert False, "recordist name missing from yaml file"
        if 'email' not in metadata['recordist']:
            assert False, "recordist email missing from yaml file"

    if 'location' not in metadata:
        assert False, "'location' fields missing from yaml file"
    else:
        if 'name' not in metadata['location']:
            assert False, "location name missing from yaml file"
        if 'latitude' not in metadata['location']:
            assert False, "location latitude missing from yaml file"
        if 'longitude' not in metadata['location']:
            assert False, "location longitude missing from yaml file"
    
    if 'equipment' not in metadata:
        assert False, "'equipment' fields missing from yaml file"
    else:
        if 'microphone' not in metadata['equipment']:
            assert False, "equipment microphone missing from yaml file"
        if 'recorder' not in metadata['equipment']:
            assert False, "equipment recorder missing from yaml file"
        if 'accessories' not in metadata['equipment']:
            assert False, "equipment accessories missing from yaml file"            
    
    if 'recording_session' not in metadata:
        assert False, "'recording_session' fields missing from yaml file"
    else:
        if 'start_time_utc' not in metadata['recording_session']:
            assert False, "recording session start time (in UTC) missing from yaml file"

if __name__ == '__main__':
    main()
