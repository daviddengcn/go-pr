/*
	A pattern recognition package.

	Gaussian classifier.
*/
package pr

/*
	LabeledFeatureSet represents a set of features collected by labels.
	Labels are from 0 to LabelCount-1
*/
type LabeledFeatureSet interface {
	// The dimention of the feature
	Dim() int
	// The number of labels
	LabelCount() int
	// The number of features for a speicified label. Labels are from 0..LabelCount-1
	FeatureCount(label int) int
	// Fetch a feature of specified label, and index. The function fill the
	// content of feature
	FetchFeature(label, index int, x []float64)
}

/*
	A Classifier can classify a feature with trained model.
*/
type Classifier interface {
	// Classify classifies the feature x and returns the label
	Classify(x []float64) int
}

/*
	A Trainer can train a Classifier given a LabeledFeatureSet.
*/
type Trainer interface {
	// Trains trains a Classifier given a LabeledFeatureSet.
	Train(featureSet LabeledFeatureSet) Classifier
}
