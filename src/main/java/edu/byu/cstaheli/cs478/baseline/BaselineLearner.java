package edu.byu.cstaheli.cs478.baseline;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import edu.byu.cstaheli.cs478.toolkit.learner.SupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

/**
 * For nominal labels, this model simply returns the majority class. For
 * continuous labels, it returns the mean value.
 * If the learning model you're using doesn't do as well as this one,
 * it's time to find a new learning model.
 */
public class BaselineLearner extends SupervisedLearner
{
    private double[] m_labels;

    public BaselineLearner()
    {
        super();
    }

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix labels = strategy.getTrainingLabels();
        m_labels = new double[labels.cols()];
        for (int i = 0; i < labels.cols(); i++)
        {
            if (labels.valueCount(i) == 0)
            {
                m_labels[i] = labels.columnMean(i); // continuous
            }
            else
            {
                m_labels[i] = labels.mostCommonValue(i); // nominal
            }
        }
    }

    public void predict(double[] features, double[] labels) throws Exception
    {
        System.arraycopy(m_labels, 0, labels, 0, m_labels.length);
    }
}
