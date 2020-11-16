import pandas as pd
import glob
from google.colab import drive

DATA_PATH = 'drive/My Drive/02456-Deep-Learning-Project/Data/'
MODEL_PATH = 'drive/My Drive/02456-Deep-Learning-Project/Models/'


def mount_drive():
    drive.mount('/content/drive')

def create_data_file(file_name):
    f = open(DATA_PATH+file_name, "w")
    f.close()
    return


def add_to_data_file(string, file_name):
    f = open(DATA_PATH + file_name, "a")
    f.write(string)
    f.close()
    return

def create_index_table_from_txt_files():
    """
        Creates a combined table of all the current runs available in the Google Drive 
    """
    all_txt_files = glob.glob(DATA_PATH +'*.txt')
    final_df = pd.DataFrame()
    
    for file in all_txt_files:
        df=pd.read_csv(file)
        df = df.set_index('Parameter name')
        df = df.transpose()
        final_df = final_df.append(df)
    return update_index_file_with_result(final_df)

#TODO: Is this the right result?
def update_index_file_with_result(df):
    """
        Add results from the corresponding .csv files
    """
    df['Last Mean Reward'] = ""
    for i in range(len(df)):
        name = df['file_name'][i].strip()
        #read csv file at DATA_PATH with current filname
        f = open(DATA_PATH + name +'.csv', "r")
        for last_line in f:
            pass
        f.close()

        _, reward = last_line.split(",") 
        #add to table
        df['Last Mean Reward'][i] = reward
    return df