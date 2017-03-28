package edu.byu.cstaheli.cs478.other;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 3/19/2017.
 */
public class Other
{
    public static void assertNumbersEqualWithEpsilon(double expected, double actual, double epsilon)
    {
        assertNumberBetween(actual, expected - epsilon, expected + epsilon);
    }

    public static void assertNumberBetween(double number, double lowerBound, double upperBound)
    {
        assertTrue(Double.compare(number, lowerBound) != -1 && Double.compare(number, upperBound) != 1, String.format("Actual: %s. Expected Bounds[%s, %s].", number, lowerBound, upperBound));
        //assertTrue(number >= lowerBound && number <= upperBound, String.format("Actual: %s. Expected Bounds[%s, %s].", number, lowerBound, upperBound));
    }
}
