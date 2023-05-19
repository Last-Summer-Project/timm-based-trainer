from multiprocessing import cpu_count

class Config:
    # ---- START OK TO EDIT ----
    dataName = None
    downloadURL = None

    modelName = "mobilenetv3_small_050.lamb_in1k"
    # ---- END OK TO EDIT ----

    batchSize = 512
    autoBatch = False
    batchMaxTries = 8
    epochs = 50
    rootDir = f"/content/drive/MyDrive/last_summer/{modelName}/"
    dataDir = "../datasets"
    shuffle = True
    numClasses = 3
    numWorkers = cpu_count() # 1
    remapClass = {
        0: 0,
        9: 1,
        10: 2
    }