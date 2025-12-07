import pathlib
import tensorflow as tf
import pandas as pd 
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator



def pre_process():
    img_path = pathlib.Path("Data/raw_data/Train/")
    #images = img_path.rglob('*.png')
    #img_output_path = "Data/processed_data/"
    #keep_going = True
    post_split = pd.read_csv("/home/panda/projects/german-street-sign/Data/raw_data/post_split.csv")
    train_df = post_split[post_split['split'] == "train"]
    val_df = post_split[post_split['split'] == "val"]
    #y_train_label = train_df["ClassId"].to_list()                                        # path to image  x-image y-label
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["ClassId"] = train_df["ClassId"].astype(str)
    val_df["ClassId"] = val_df["ClassId"].astype(str)
    test_df = pd.read_csv("/home/panda/projects/german-street-sign/Data/raw_data/Test.csv")
    test_df = test_df.copy()
    test_df["ClassId"] = test_df["ClassId"].astype(str)
    





    #print(y_train_label)

    '''
    for item in dir_path.glob('**/'):
        #print(item)
        for image in item.glob('*'):
            print(image)

    for image in images:     # This will load all images. Pre-processing can start.
        print(image)

        for image_path in images:
        if keep_going == True:
            data = tf.io.read_file(str(image_path))                       # should read image
            img = tf.io.decode_png(data,channels = 3)                     # format for tf ?
            img = tf.image.resize(img, [60,60])                           # resize to 60x60 pixels
            img = tf.cast(img, tf.float32)/255.0                          # Normalise to 0-1
            
            keep_going = False
    '''
    train_gen = ImageDataGenerator(
                rescale = 1.0/255.0,
                rotation_range = 5,
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                brightness_range = (0.9,1.1))
    val_gen = ImageDataGenerator(
                    rescale=1.0/255.0)
    test_gen = ImageDataGenerator(
                    rescale=1.0/255.0)

    train_ds = train_gen.flow_from_dataframe(                                 # Training Dataset
                    dataframe=train_df,
                    directory="/home/panda/projects/german-street-sign/Data/raw_data",
                    x_col="Path",
                    y_col="ClassId",
                    target_size=(60,60),
                    color_mode="rgb",
                    class_mode="categorical",             # catergorical if str, set to sparse if int.
                    batch_size=64,
                    shuffle=True)

    val_ds = val_gen.flow_from_dataframe(                                     # Validation Dataset
                    dataframe=val_df,
                    directory="/home/panda/projects/german-street-sign/Data/raw_data",
                    x_col="Path",
                    y_col="ClassId",
                    target_size=(60,60),
                    color_mode="rgb",
                    class_mode="categorical",              # catergorical if str, set to sparse if int.
                    batch_size=64,
                    shuffle=False)
    
    test_ds = val_gen.flow_from_dataframe(                                     # Validation Dataset
                    dataframe=test_df,
                    directory="/home/panda/projects/german-street-sign/Data/raw_data",
                    x_col="Path",
                    y_col="ClassId",
                    target_size=(60,60),
                    color_mode="rgb",
                    class_mode="categorical",              # catergorical if str, set to sparse if int.
                    batch_size=64,
                    shuffle=False)
    
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    pre_process()