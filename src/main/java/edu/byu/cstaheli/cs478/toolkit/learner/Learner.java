package edu.byu.cstaheli.cs478.toolkit.learner;

/**
 * Created by cstaheli on 4/2/2017.
 */
public class Learner
{
    private String outputFile;

    protected boolean shouldOutput()
    {
        return (getOutputFile() != null);
    }

    protected String getOutputFile()
    {
        return outputFile;
    }

    public void setOutputFile(String outputFile)
    {
        this.outputFile = outputFile;
    }
}
