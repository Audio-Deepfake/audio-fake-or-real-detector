#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:18:12 2021

@author: jass
"""
import json
import pandas as pd

OUTPUT_JSON = "output.json"
OUTPUT_CSV = "output.csv"

id_to_text = {
    0:"fake",
    1:"real"
    }

def segments_to_full_converter(op_data, meta_data):
    output = {}
    i = 0
    while i<len(op_data):
        j = 0
        patch_arr = []
        text_label_arr =[]
        for segment,file_name in zip(meta_data["segments"],meta_data["file_name"]):
            j += segment
            patch = op_data[i:j]
            patch_max=max(patch,key=patch.count)
            patch_arr.append(patch_max)
            text_label_arr.append(id_to_text[patch_max])
            i += segment

    output["file_name"] = meta_data["file_name"]
    output["aud_type"] = text_label_arr
    output["label"] = patch_arr
    print(output)
    create_json(output, OUTPUT_JSON)
    create_csv(output, OUTPUT_CSV)
    
def create_json(data, output_fp):
    
    with open(output_fp, "w") as fp:
        json.dump(data,fp, indent = 4)

def create_csv(data,output_fp):
    s1 = pd.Series(data["file_name"], name = "file_name")
    s2 = pd.Series(data["aud_type"], name = "aud_type")
    s3 = pd.Series(data["label"], name = "label")
    
    df = pd.concat([s1,s2,s3], axis = 1)
    df.to_csv(output_fp, index= False)

if __name__ == "__main__":
 
    # The op_data is the data which is given as a output by the model after predicting the segmented user input audio. The sample data here is derived from 61 segments of 21 audio files which is given as a input by the user
    op_data = {
        "labels":[1,0,0,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,1,1,0,1,1,0,1,0,1,0,1,0,1,0]
        }
    with open("meta_data.json","r") as fp:
        meta_data = json.load(fp)
    with open("feature_data.json","r") as fp:
        data = json.load(fp)
    segments_to_full_converter(op_data["labels"],meta_data)
