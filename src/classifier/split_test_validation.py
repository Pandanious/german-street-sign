import pandas as pd
from sklearn.model_selection import train_test_split

def split_validation_test() -> pd.DataFrame: 
    train_data = pd.read_csv("Data/raw_data/Train.csv")    # should be a Dataframe

    #print(train_data.info)
    path = train_data["Path"]   # x is path usually
    label = train_data["ClassId"] # y is class


    x_train,x_val,y_train,y_val = train_test_split(path,label,test_size=0.2,stratify=label,random_state=41)
    #train_split = pd.DataFrame({"Path":x_train,"ClassId":y_train}).assign(split= "train")
    #val_split = pd.DataFrame({"Path":x_val,"ClassId":y_val}).assign(split= "val")
    #train_split = pd.DataFrame({"Path":x_train,"ClassId":y_train}).assign(split= "train")
    #post_split = pd.concat([train_split, val_split], ignore_index=True)
    #post_split.to_csv("Data/raw_data/post_split.csv", index=False)
    train_data = train_data.copy()
    #print(train_data.info)

    train_data.loc[x_train.index, "split"] = "train"
    train_data.loc[x_val.index, "split"] = "val"



    train_data.to_csv("Data/raw_data/post_split.csv", index=False)

    return train_data


if __name__ == "__main__":
    split_validation_test()