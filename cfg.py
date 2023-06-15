from multiprocessing import cpu_count


class Config:
    # ---- START OK TO EDIT ----
    dataName = "sangchu-noaug-resize_div4"
    downloadURL = "https://last-summer-r2.won-jung.kim/sangchu-noaug-resize_div4.zip"

    modelName = "mobilenetv3_small_050.lamb_in1k"
    # ---- END OK TO EDIT ----

    # -- SETUP --
    batchSize = 4
    autoBatch = False
    batchMaxTries = 8
    epochs = 50
    rootDir = f"./"
    dataDir = "./datasets"
    shuffle = True
    earlyStoppingPatience = 10

    # -- DATA --
    numClasses = 3
    numWorkers = 1  # 1
    remapClass = {0: 0, 9: 1, 10: 2}

    # -- OPTIONS --
    optimizer = "adam"
    learningRate = 1e-3
    transfer = True
    tuneFcOnly = True
    exportable = True
    scriptable = True
    saveHyperParam = True