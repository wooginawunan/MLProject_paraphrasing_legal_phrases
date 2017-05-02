#module load python/gnu/2.7.11
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import GradientBoostedTrees
from csv import reader
# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line]
    return LabeledPoint(values[-1] if values[-1]==1 else 0, values[:-1])

data = sc.textFile('mergedAB_delete_all_empty.csv')
data = data.mapPartitions(lambda x: reader(x))
header = data.first()
data = data.filter(lambda x: x != header)
#data = data.filter(lambda x: x[-1] in ['1', '-1'])

parsedData = data.map(parsePoint)
instance_count = float(parsedData.count())

# Build the SVM model
model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / instance_count
print("Training Error = " + str(trainErr))

#build the Logistic regression model 
model_lr = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds_lr = parsedData.map(lambda p: (p.label, model_lr.predict(p.features)))
trainErr = labelsAndPreds_lr.filter(lambda (v, p): v != p).count() / instance_count
print("Training Error = " + str(trainErr))



model_lr = LogisticRegressionWithSGD.train(parsedData)
labelsAndPreds_lr = parsedData.map(lambda p: (p.label, model_lr.predict(p.features)))
trainErr = labelsAndPreds_lr.filter(lambda (v, p): v != p).count() / instance_count
print("Training Error = " + str(trainErr))

# Gradient Boosting Model
model_GB = GradientBoostedTrees.trainClassifier(parsedData, {}, numIterations=50)
print(model_GB)

labelsAndPreds_GB = parsedData.map(lambda p: (p.label, model_GB.predict(p.features)))
trainErr = labelsAndPreds_GB.filter(lambda (v, p): v != p).count() / instance_count
print("Training Error = " + str(trainErr))

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=2, 
                                     impurity='gini', maxDepth=20, maxBins=32)


# Save and load model
model.save(sc, "SVM_nomal")
sameModel = SVMModel.load(sc, "SVM_nomal")

# Instantiate metrics object
metrics = BinaryClassificationMetrics(labelsAndPreds)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
