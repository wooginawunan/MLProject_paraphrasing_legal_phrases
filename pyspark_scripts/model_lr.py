from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def parsePoint(line):
    values = [float(x) for x in line]
    return LabeledPoint(values[-1] if values[-1]==1 else 0, values[:-1])

data = sc.textFile('mergedAB_delete_all_empty.csv')
data = data.mapPartitions(lambda x: reader(x))
#header = data.first()
#data = data.filter(lambda x: x != header)
data = data.filter(lambda x: x[-1] in ['1', '-1'])

parsedData = data.map(parsePoint)


# Build the model
model_lr = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds_lr = parsedData.map(lambda p: (p.label, model_lr.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "myModelPath")
sameModel = LogisticRegressionModel.load(sc, "myModelPath")