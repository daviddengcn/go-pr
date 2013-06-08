package pr

import (
	"github.com/skelterjohn/go.matrix"
	"math"
)

/*
	A *GaussianClassifier use the multivariate Gaussian distribution as the
	likelyhood function.
	(wikipedia: http://en.wikipedia.org/wiki/Multivariate_normal_distribution)

	  f(x) = (1/(sqrt((2*Pi)^k*det(Sigma)))*exp(-1/2*(x-mu)^T*inv(Sigma)*(x-mu)),

	where k is the dimention of feature x, Sigma is the covariance matrix:

	  Sigma = [Cov[Xi, Xj]],

	and mu is the mean vector. det, exp and inv are determinant/exponent/inverse
	functions, respectively.
*/
type GaussianClassifier struct {
	// the means
	Means [][]float64
	// the inverse matrix of Sigma times -1/2
	Precs [][]float64
	// logarithm coefficents.(log(1/(sqrt((2*Pi)^k*det(Sigma))))
	LogCoefs []float64
	// if non-nil, the logarithm of prior priorities
	LogPrior []float64
}

/*
	SetPrior sets the prior probabilities of all labels.
*/
func (gc *GaussianClassifier) SetPrior(priors []float64) {
	if gc.LogPrior == nil {
		gc.LogPrior = make([]float64, len(priors))
	}
	for i := range priors {
		gc.LogPrior[i] = math.Log(priors[i])
	}
}

// Implementation of Classifier.Classify
func (gc *GaussianClassifier) Classify(x []float64) int {
	bestLogP := 0.
	bestLabel := -1

	for lbl := range gc.LogCoefs {
		logP := gc.LogPosterior(lbl, x)

		if bestLabel < 0 || logP > bestLogP {
			bestLabel, bestLogP = lbl, logP
		}
	}

	return bestLabel
}

/*
	LogLikelyhood returns the logarithm of the likelyhood of the feature x on a
	specified label.
*/
func (gc *GaussianClassifier) LogLikelyhood(label int, x []float64) float64 {
	logP := gc.LogCoefs[label]

	mean := gc.Means[label]
	prec := gc.Precs[label]

	dim := len(mean)

	/* logP += (x -  mu)' * Sigma * (x - mu) */
	for k := range mean {
		vk := x[k] - mean[k]
		for l := range mean {
			vl := x[l] - mean[l]
			logP += vk * vl * prec[k*dim+l]
		}
	}

	return logP
}

/*
	LogPosterior returns the logarithm of the posterior probability of a feature
	on a specified label.
*/
func (gc *GaussianClassifier) LogPosterior(label int, x []float64) float64 {
	if gc.LogPrior == nil {
		return gc.LogLikelyhood(label, x)
	}
	return gc.LogLikelyhood(label, x) + gc.LogPrior[label]
}

/*
	The trainer for a Gaussian classifier
*/
type GaussianTrainer struct {
}

/*
	GaussianTrain trains a *GaussianClassifier from a LabeledFeatureSet.
*/
func GaussianTrain(lfs LabeledFeatureSet) *GaussianClassifier {
	lblCnt := lfs.LabelCount()
	dim := lfs.Dim()
	clsfr := &GaussianClassifier{
		Means:    make([][]float64, lblCnt),
		Precs:    make([][]float64, lblCnt),
		LogCoefs: make([]float64, lblCnt),
	}

	x := make([]float64, dim)

	sigma := make([]float64, dim*dim)
	for lbl := range clsfr.Means {
		mean := make([]float64, dim)

		cnt := lfs.FeatureCount(lbl)

		for i := 0; i < cnt; i++ {
			lfs.FetchFeature(lbl, i, x)
			for k := range x {
				mean[k] += x[k]
			}
		}

		for k := range mean {
			mean[k] /= float64(cnt)
		}

		for i := range sigma {
			sigma[i] = 0.
		}
		for i := 0; i < cnt; i++ {
			lfs.FetchFeature(lbl, i, x)
			for k := 0; k < dim; k++ {
				for l := k; l < dim; l++ {
					sigma[k*dim+l] += (x[k] - mean[k]) * (x[l] - mean[l])
				}
			}
		}
		if cnt > 1 {
			for i := range sigma {
				sigma[i] /= float64(cnt - 1)
			}
		}
		// copy the left-bottom part from right-top part
		for k := 0; k < dim; k++ {
			for l := 0; l < k; l++ {
				sigma[k*dim+l] = sigma[l*dim+k]
			}
		}

		mat := matrix.MakeDenseMatrix(sigma, dim, dim)
		inv, err := mat.Inverse()
		if err != nil {
			return nil
		}

		inv.Scale(-0.5)

		det := mat.Det()

		clsfr.Means[lbl] = mean
		clsfr.Precs[lbl] = inv.Array()
		clsfr.LogCoefs[lbl] = -0.5 * (math.Log(2.*math.Pi)*float64(dim) + math.Log(det))
	}

	return clsfr
}

// Implementation of Trainer.Train
func (gt *GaussianTrainer) Train(lfs LabeledFeatureSet) Classifier {
	return GaussianTrain(lfs)
}
