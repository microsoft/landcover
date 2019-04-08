import io  
import shutil  
import time
import os
from azure.storage.blob import BlockBlobService
import uuid
import azure_config as cfg

temp_path = "azure-temp-storage"

def delete_temp_folders():
    now = time.time()
    old = now - 1 * 24 * 60 * 60
    
    for root, dirs, files in os.walk(temp_path, topdown=False):
        for dir in dirs:
            if os.stat(os.path.join(os.getcwd(),temp_path, dir)).st_mtime < old: 
                delete_temp_folder(dir)

def delete_temp_folder(folder_name):
    try:
        path = os.path.join(os.getcwd(),temp_path, folder_name)
        shutil.rmtree(path)
    except Exception as e:
        print("Error occurred while deleting temp azure folder: " + str(e))

def get_blob(container_name, blob_name):
    
    delete_temp_folders()
    
    if(container_name == 'esri-naip'):
        block_blob_service = BlockBlobService(account_name=cfg.mslandcoverstorageeast['account_name'], account_key=cfg.mslandcoverstorageeast["account_key"])
    else:
        block_blob_service = BlockBlobService(account_name=cfg.modeloutput["account_name"], account_key=cfg.modeloutput["account_key"])

    file_name = blob_name.split("/").pop(-1)
    temp_sub_folder = str(uuid.uuid4())
    full_temp_folder = 'azure-temp-storage/' + temp_sub_folder
    file_path = full_temp_folder + "/" + file_name

    os.mkdir(full_temp_folder)

    if(file_name.endswith("mrf")):
        block_blob_service.get_blob_to_path(container_name, blob_name, file_path)

        #download idx file to temp folder
        idx_path = file_path.replace("mrf", "idx")
        idx_blob_name = blob_name.replace("mrf", "idx")
        block_blob_service.get_blob_to_path(container_name, idx_blob_name, idx_path)
        
        #download lrc file to temp folder
        lrc_path = file_path.replace("mrf", "lrc")
        lrc_blob_name = blob_name.replace("mrf", "lrc")
        block_blob_service.get_blob_to_path(container_name, lrc_blob_name, lrc_path)

    elif(file_name.endswith("tif")):
        block_blob_service.get_blob_to_path(container_name, blob_name, file_path)


    return file_path, temp_sub_folder