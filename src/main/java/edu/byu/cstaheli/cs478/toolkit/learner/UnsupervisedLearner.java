package edu.byu.cstaheli.cs478.toolkit.learner;

import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

/**
 * Created by cstaheli on 3/28/2017.
 */
public abstract class UnsupervisedLearner
{
    public abstract void cluster(Matrix dataset);
}
