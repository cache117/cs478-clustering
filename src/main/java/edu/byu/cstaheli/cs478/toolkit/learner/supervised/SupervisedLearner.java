package edu.byu.cstaheli.cs478.toolkit.learner.supervised;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import edu.byu.cstaheli.cs478.toolkit.learner.Learner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

public abstract class SupervisedLearner extends Learner
{

    protected SupervisedLearner()
    {
    }

    public abstract void train(LearningStrategy strategy) throws Exception;

    protected double getBestAccuracy(double newValue, double previousBest)
    {
        if (newValue > previousBest)
        {
            previousBest = newValue;
        }
        return previousBest;
    }

    // The model must be trained before you call this method. If the label is nominal,
    // it returns the predictive accuracy. If the label is continuous, it returns
    // the root mean squared error (RMSE). If confusion is non-NULL, and the
    // output label is nominal, then confusion will hold stats for a confusion matrix.
    public double measureAccuracy(Matrix features, Matrix labels, Matrix confusion) throws Exception
    {
        if (features.rows() != labels.rows())
        {
            throw (new Exception("Expected the features and labels to have the same number of rows"));
        }
        if (labels.cols() != 1)
        {
            throw (new Exception("Sorry, this method currently only supports one-dimensional labels"));
        }
        if (features.rows() == 0)
        {
            throw (new Exception("Expected at least one row"));
        }

        int labelValues = labels.valueCount(0);
        if (labelValues == 0) // If the label is continuous...
        {
            // The label is continuous, so measure root mean squared error
            double[] prediction = new double[1];
            double sse = 0.0;
            for (int i = 0; i < features.rows(); i++)
            {
                double[] feat = features.row(i);
                double[] targ = labels.row(i);
                prediction[0] = 0.0; // make sure the prediction is not biased by a previous prediction
                predict(feat, prediction);
                double delta = targ[0] - prediction[0];
                sse += (delta * delta);
            }
            return Math.sqrt(sse / features.rows());
        }
        else
        {
            // The label is nominal, so measure predictive accuracy
            if (confusion != null)
            {
                confusion.setSize(labelValues, labelValues);
                confusion.setDatasetName(features.getDatasetName());
                for (int i = 0; i < labelValues; i++)
                {
                    confusion.setAttrName(i, labels.attrValue(0, i));
                }
            }
            int correctCount = 0;
            double[] prediction = new double[1];
            for (int i = 0; i < features.rows(); i++)
            {
                double[] feat = features.row(i);
                int targ = (int) labels.get(i, 0);
                if (targ >= labelValues)
                {
                    throw new Exception("The label is out of range");
                }
                predict(feat, prediction);
                int pred = (int) prediction[0];
                if (confusion != null)
                {
                    confusion.set(targ, pred, confusion.get(targ, pred) + 1);
                }
                if (pred == targ)
                {
                    correctCount++;
                }
            }
            return (double) correctCount / features.rows();
        }
    }

    // A feature vector goes in. A label vector comes out. (Some supervised
    // learning algorithms only support one-dimensional label vectors. Some
    // support multi-dimensional label vectors.)
    public abstract void predict(double[] features, double[] labels) throws Exception;

    protected double calcMeanSquaredError(Matrix features, Matrix labels) throws Exception
    {
        assert features.rows() == labels.rows();
        double mse = 0;
        for (int i = 0; i < features.rows(); ++i)
        {
            double[] row = features.row(i);
            double output = labels.get(i, 0);
            double[] label = new double[1];
            predict(row, label);
            double predicted = label[0];
            mse += calcSquaredError(output, predicted);
        }
        return mse / features.rows();
    }

    private double calcSquaredError(double expected, double actual)
    {
        return (expected - actual) * (expected - actual);
    }

}
