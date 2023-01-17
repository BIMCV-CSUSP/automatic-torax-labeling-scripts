import preprocess
import cputils

import multiprocessing
import os
import cv2
import sys
import pandas


def preprocess_list(path_list, monochrome_list, store_path):
    file_list = []
    sys.stdout.flush()
    for file_path, monochrome in zip(path_list, monochrome_list):
        sys.stdout.flush()
        image = preprocess.preprocess(file_path, monochrome)
        if image is None:
            file_list.append(None)
        else:
            filename = file_path.split('/')[-1]
            new_file_path = os.path.join(store_path, filename)
            cv2.imwrite(new_file_path, image)
            file_list.append(new_file_path)
    

    return file_list


if __name__ == '__main__':
    datapath = sys.argv[1]
    files_per_batch = int(sys.argv[2])
    scratch_folder = sys.argv[3]
    return_folder = sys.argv[4]
    out_path = sys.argv[5]
    
    df = pandas.read_csv(datapath, sep='\t')
    
    processed_folder = os.path.join(scratch_folder,"processed")
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    originals_folder = os.path.join(scratch_folder,"originals")
    if not os.path.exists(originals_folder):
        os.makedirs(originals_folder)
    
    num_batches = len(df)//files_per_batch
    if len(df) % files_per_batch != 0:
        num_batches += 1
    
    df_act = pandas.DataFrame()
    df_ant = pandas.DataFrame()
    df_next= df[:files_per_batch].copy()
    
    new_paths = cputils.copy_list(list(df_next['png']), originals_folder)
    df_next['temp_path'] = new_paths
    
    new_df = pandas.DataFrame()
    print(num_batches)

    with multiprocessing.Pool(processes=3) as pool:
        for i in range(num_batches):
            df_ant = df_act.copy()
            df_act = df_next.copy()
            df_next= df[files_per_batch*(i+1):files_per_batch*(i+2)].copy()
            
            res_copy_next = pool.apply_async(cputils.copy_list, (list(df_next['png']), originals_folder))
            print(f'running step {i} with {len(df_ant)=}, {len(df_act)=}, {len(df_next)=}')
            for i,file_path in zip(df_act.index, df_act['png']):
                print(f"image {i} with path {file_path}")
            sys.stdout.flush()
            res_preprocess = pool.apply_async(preprocess_list, (list(df_act['temp_path']), list(df_act['Photometric Interpretation']), processed_folder))
            if len(df_ant) > 0:
                res_copy_back = pool.apply_async(cputils.copy_list, (list(df_ant['temp_path']),return_folder))
                new_paths = res_copy_back.get()
                df_ant['preprocessed_path'] = new_paths
                for p in list(df_ant['temp_path']):
                    if p:
                        os.remove(p)
            
            new_paths = res_copy_next.get()
            df_next['temp_path'] = new_paths
            
            new_paths = res_preprocess.get()
            for p in list(df_act['temp_path']):
                if p:
                    os.remove(p)
            df_act['temp_path'] = new_paths   
            
            new_df = new_df.append(df_ant)

    new_paths = cputils.copy_list(list(df_act['temp_path']),return_folder)
    df_act['preprocessed_path'] = new_paths
    for p in list(df_act['temp_path']):
        if p:
            os.remove(p)
    new_df = new_df.append(df_act)

    new_df.to_csv(out_path, sep="\t", index=False)

    
