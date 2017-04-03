package edu.byu.cstaheli.cs478.toolkit.learner.unsupervised;

import edu.byu.cstaheli.cs478.toolkit.learner.Learner;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

/**
 * Created by cstaheli on 3/28/2017.
 */
public abstract class UnsupervisedLearner extends Learner
{
    private boolean isVerbose;

    public UnsupervisedLearner()
    {
        isVerbose = false;
    }

    public abstract void cluster(Matrix dataset);

    public void setIsVerbose(boolean verbose)
    {
        isVerbose = verbose;
    }
}
