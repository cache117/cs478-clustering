package edu.byu.cstaheli.cs478.nearest_neighbor;

import edu.byu.cstaheli.cs478.toolkit.exception.MatrixException;
import edu.byu.cstaheli.cs478.toolkit.learner.supervised.SupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.io.FileWriter;
import java.util.Map;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.euclideanDistance;
import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.squaredDistance;

/**
 * Created by cstaheli on 3/14/2017.
 */
public class NearestNeighbor extends SupervisedLearner
{
    private int numberOfNeighborsToCompareTo;
    private Matrix trainingData;
    private boolean useDistanceWeighting;
    private boolean useRegression;

    public NearestNeighbor()
    {
        this.numberOfNeighborsToCompareTo = 3;
        useDistanceWeighting = false;
    }

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        strategy.setUseValidationSet(false);
        //Allows data to be shuffled if needed
//        strategy.getTrainingFeatures();
//        strategy.getTrainingLabels();
        trainingData = strategy.getTrainingData();
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        Neighbors neighbors = new Neighbors(numberOfNeighborsToCompareTo, useDistanceWeighting, useRegression);
        for (int i = 0; i < trainingData.rows(); ++i)
        {
            double[] row = trainingData.row(i);
            double distance = distance(row, features);
            neighbors.addPotential(row, distance);
        }
        labels[0] = neighbors.predict();
    }

    /**
     * Returns the distance between the two vectors. Note that the existing row <i>can</i> include it's output class,
     * but doesn't need to. The sizes of the two input vectors must either be the same, or else the existing row must
     * have one more element.
     *
     * @param existingRow   a row from the existing training data.
     * @param predictionRow the new row to compare to.
     * @return the distance between the two vectors
     * @throws MatrixException if any features are nominal and the matrix doesn't line up with the given rows.
     */
    protected double distance(double[] existingRow, double[] predictionRow) throws MatrixException
    {
        // Existing row should have 1 more column (the output)
        assert existingRow.length == predictionRow.length + 1 || existingRow.length == predictionRow.length;
        double distance = 0;
        for (int i = 0; i < predictionRow.length; ++i)
        {
            distance += distance(i, existingRow[i], predictionRow[i]);
        }
        return distance;
    }

    protected double distance(int column, double firstValue, double secondValue) throws MatrixException
    {
        if (new Double(firstValue).equals(Matrix.MISSING) || new Double(secondValue).equals(Matrix.MISSING))
        {
            return 1;
        }
        if (trainingData.valueCount(column) == 0)
        {
            return euclideanDistance(firstValue, secondValue);
        }
        else
        {
            if (trainingData.valueCount(trainingData.cols() - 1) != 0)
            {
                return valueDistanceMetric(column, firstValue, secondValue);
            }
            else
            {
                return euclideanDistance(firstValue, secondValue);
            }
        }
    }

    protected double valueDistanceMetric(int column, double firstValue, double secondValue) throws MatrixException
    {
        double valueDistanceMetric = 0;
        Map<Double, Integer> outputOccurrences = trainingData.getColumnOccurrences(trainingData.cols() - 1);
        Matrix hasFirstValue = trainingData.getRowsWithColumnClass(column, firstValue);
        //# Times attribute a had value x
        double nax = hasFirstValue.rows();
        Matrix hasSecondValue = trainingData.getRowsWithColumnClass(column, secondValue);
        //# Times attribute a had value y
        double nay = hasSecondValue.rows();
        for (Map.Entry<Double, Integer> entry : outputOccurrences.entrySet())
        {
            Matrix hasFirstAndOutput = hasFirstValue.getRowsWithColumnClass(hasFirstValue.cols() - 1, entry.getKey());
            //# times attribute a=x and class was c
            double naxc = hasFirstAndOutput.rows();
            Matrix hasSecondAndOutput = hasSecondValue.getRowsWithColumnClass(hasSecondValue.cols() - 1, entry.getKey());
            //# times attribute a=y and class was c
            double nayc = hasSecondAndOutput.rows();
            double paxc = naxc / nax;
            double payc = nayc / nay;
            valueDistanceMetric += squaredDistance(paxc, payc);
        }
        return valueDistanceMetric;
    }

    public int getNumberOfNeighborsToCompareTo()
    {
        return numberOfNeighborsToCompareTo;
    }

    public void setNumberOfNeighborsToCompareTo(int numberOfNeighborsToCompareTo)
    {
        this.numberOfNeighborsToCompareTo = numberOfNeighborsToCompareTo;
    }

    public Matrix getTrainingData()
    {
        return trainingData;
    }

    public void setTrainingData(Matrix trainingData)
    {
        this.trainingData = trainingData;
    }

    public boolean isUsingDistanceWeighting()
    {
        return useDistanceWeighting;
    }

    public void setUseDistanceWeighting(boolean useDistanceWeighting)
    {
        this.useDistanceWeighting = useDistanceWeighting;
    }

    public void outputFinalStatistics(double testingAccuracy)
    {
        if (shouldOutput())
        {
            try (FileWriter writer = new FileWriter(getOutputFile(), true))
            {
                writer.append(String.format("%s, %s\n", numberOfNeighborsToCompareTo, testingAccuracy));
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
    }

    public boolean isUseRegression()
    {
        return useRegression;
    }

    public void setUseRegression(boolean useRegression)
    {
        this.useRegression = useRegression;
    }
}
