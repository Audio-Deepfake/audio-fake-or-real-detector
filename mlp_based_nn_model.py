
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import classification_report 
from sklearn import metrics
import matplotlib.pyplot as plt

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "feature_dataset_2.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    
    X = np.array(data["spect"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y, data

np.random.seed(45)

def model():
        # load data
    X, y, data = load_data(DATA_PATH)
    
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(2, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
    print(history)

#%%
    seq_predictions_total=model.predict(X_test)
    print(seq_predictions_total.shape)
    seq_predictions_total=np.transpose(seq_predictions_total)[1]
    print(seq_predictions_total.shape) 
    # Applying transformation to get binary values predictions with 0.5 as thresold
    seq_predictions = list(map(lambda x: 0 if x<0.5 else 1, seq_predictions_total))
    seq_predictions = np.array(seq_predictions)
#%%
    # add some other stats based on confusion matrix
    clf_report = classification_report(y_test, seq_predictions)
    print(clf_report)
     # auc information
    fpr, tpr, _ = metrics.roc_curve(y_test,  seq_predictions_total)
    auc = metrics.roc_auc_score(y_test, seq_predictions_total)
    
    metrics_var = []
    metrics_var.append(fpr)
    metrics_var.append(tpr)
    metrics_var.append(auc)
    metrics_var.append("Multilayer Perceptron")
    # log loss
    log_loss = metrics.log_loss(y_test, seq_predictions_total)
    plt.plot(fpr, tpr, label="{}, AUC={:.3f}".format("Multi layer perceptron", auc))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    
    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    
    plt.savefig("roc-auc-mlp",dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_var
if __name__ == "__main__":
    
    meta = model()