package edu.byu.cstaheli.cs478.toolkit.utility;

/**
 * Class for parsing out the command line arguments
 */
public class ArgParser
{

    private String arff;
    private String learner;
    private String evaluation;
    private String evalExtra;
    private boolean verbose;
    private boolean normalize;
    private String wantedIndex;

    //You can add more options for specific learning models if you wish
    public ArgParser(String[] argv)
    {
        try
        {
            for (int i = 0; i < argv.length; i++)
            {
                switch (argv[i])
                {
                    case "-V":
                        verbose = true;
                        break;
                    case "-N":
                        normalize = true;
                        break;
                    case "-A":
                        arff = argv[++i];
                        break;
                    case "-L":
                        learner = argv[++i];
                        break;
                    case "-W":
                        wantedIndex = argv[++i];
                        break;
                    case "-E":
                        evaluation = argv[++i];
                        switch (argv[i])
                        {
                            case "static":
                                //expecting a test set name
                                evalExtra = argv[++i];
                                break;
                            case "random":
                                //expecting a double representing the percentage for testing
                                //Note stratification is NOT done
                                evalExtra = argv[++i];
                                break;
                            case "cross":
                                //expecting the number of folds
                                evalExtra = argv[++i];
                                break;
                            case "training":
                                break;
                            case "cluster":
                                evalExtra = argv[++i];
                                break;
                            default:
                                System.out.println("Invalid Evaluation Method: " + argv[i]);
                                System.exit(0);
                        }
                        break;
                    default:
                        System.out.println("Invalid parameter: " + argv[i]);
                        System.exit(0);
                }
            }

        }
        catch (Exception e)
        {
            printUsageInstructions(false);
        }

        if (arff == null || learner == null || evaluation == null)
        {
            printUsageInstructions(true);
        }
    }

    private void printUsageInstructions(boolean printNormalized)
    {
        System.out.println("Usage:");
        System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
        System.out.println("OPTIONS:");
        System.out.println("-V Print the confusion matrix and learner accuracy on individual class values");
        if (printNormalized)
        {
            System.out.println("-N Use normalized data");
        }
        System.out.println();
        System.out.println("Possible evaluation methods are:");
        System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
        System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
        System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
        System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
        System.exit(0);
    }

    //The getter methods
    public String getARFF()
    {
        return arff;
    }

    public String getLearner()
    {
        return learner;
    }

    public String getEvaluation()
    {
        return evaluation;
    }

    public String getEvalParameter()
    {
        return evalExtra;
    }

    public boolean isVerbose()
    {
        return verbose;
    }

    public boolean isNormalized()
    {
        return normalize;
    }

    public String getWantedIndex()
    {
        return wantedIndex;
    }
}
