from multiprocessing import cpu_count


class Config:
    # ---- START OK TO EDIT ----
    dataName = None
    downloadURL = None

    modelName = "mobilenetv3_small_050.lamb_in1k"
    # ---- END OK TO EDIT ----

    # -- SETUP --
    batchSize = 512
    autoBatch = False
    batchMaxTries = 8
    epochs = 50
    rootDir = f"/content/drive/MyDrive/last_summer/{modelName}/"
    dataDir = "../datasets"
    shuffle = True

    # -- DATA --
    numClasses = 3
    numWorkers = cpu_count()  # 1
    remapClass = {0: 0, 9: 1, 10: 2}

    # -- OPTIONS --
    optimizer = "adam"
    learningRate = 1e-3
    transfer = True
    tuneFcOnly = True
    exportable = True
    scriptable = True
