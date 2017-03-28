package edu.byu.cstaheli.cs478.toolkit.learner;

import edu.byu.cstaheli.cs478.toolkit.utility.ArgParser;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.util.Random;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class LearnerData
{
    private final Random rand;
    private final ArgParser parser;
    private final Matrix arffData;

    public LearnerData(Random rand, ArgParser parser, Matrix arffData)
    {
        this.rand = rand;
        this.parser = parser;
        this.arffData = arffData;
    }

    public Random getRandom()
    {
        return rand;
    }

    public Matrix getArffData()
    {
        return arffData;
    }

    public String getEvaluation()
    {
        return parser.getEvaluation();
    }

    public String getEvalParameter()
    {
        return parser.getEvalParameter();
    }

    public boolean isVerbose()
    {
        return parser.isVerbose();
    }

    public boolean isNormalized()
    {
        return parser.isNormalized();
    }
}
