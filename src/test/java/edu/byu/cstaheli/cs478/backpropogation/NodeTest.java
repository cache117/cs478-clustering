package edu.byu.cstaheli.cs478.backpropogation;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static edu.byu.cstaheli.cs478.other.Other.assertNumberBetween;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by cstaheli on 2/16/2017.
 */
public class NodeTest
{

    @Test
    void testHardCodedProcess()
    {
        List<InputWeight> inputWeights = new ArrayList<>(2);
        inputWeights.add(new InputWeight(1));
        inputWeights.add(new InputWeight(1));
        Node node2 = new Node(inputWeights, 1);
        List<Double> inputs = new ArrayList<>(2);
        inputs.add(0d);
        inputs.add(0d);
        double net2 = node2.calcNet(inputs);
        assertEquals(1, net2);
        double output2 = node2.calcOutput(net2);
        assertNumberBetween(output2, .7305, .7314);

        inputWeights = new ArrayList<>(2);
        inputWeights.add(new InputWeight(1));
        inputWeights.add(new InputWeight(1));
        Node node3 = new Node(inputWeights, 1);
        double net3 = node3.calcNet(inputs);
        assertEquals(1, net3);
        double output3 = node3.calcOutput(net3);
        assertNumberBetween(output3, .7305, .7314);

        inputWeights = new ArrayList<>(2);
        inputWeights.add(new InputWeight(1));
        inputWeights.add(new InputWeight(1));
        Node node1 = new Node(inputWeights, 1);
        inputs = new ArrayList<>(2);
        inputs.add(output2);
        inputs.add(output3);
        double net1 = node1.calcNet(inputs);
        assertNumberBetween(net1, 2.4620, 2.4622);
        double output1 = node1.calcOutput(net1);
        assertNumberBetween(output1, .921, .922);
        double gradient1 = node1.calcGradient(output1);
        assertNumberBetween(gradient1, .0723, .0724);
        double error1 = node1.calcOutputNodeError(gradient1, 1, output1);
        assertNumberBetween(error1, .005686, .005687);

        List<Double> outputErrors = new ArrayList<>(1);
        outputErrors.add(error1);
        List<Double> weightsToOutputs = new ArrayList<>(1);
        double inputWeight2To1 = node1.getInputWeight(0);
        weightsToOutputs.add(inputWeight2To1);

        double gradient2 = node2.calcGradient(output2);
        assertNumberBetween(gradient2, .1966, .1967);
        double error2 = node2.calcHiddenNodeError(gradient2, outputErrors, weightsToOutputs);
        assertNumberBetween(error2, .00111, .00112);

        outputErrors = new ArrayList<>(1);
        outputErrors.add(error1);
        weightsToOutputs = new ArrayList<>(1);
        double inputWeight3To1 = node1.getInputWeight(1);
        weightsToOutputs.add(inputWeight3To1);

        double gradient3 = node3.calcGradient(output2);
        assertNumberBetween(gradient3, .1966, .1967);
        double error3 = node2.calcHiddenNodeError(gradient3, outputErrors, weightsToOutputs);
        assertNumberBetween(error3, .00111, .00112);

        double learningRate = 1;
        double momentum = 0;
        double deltaWeight2To1 = Node.calculateWeightDelta(learningRate, error1, output2, 0, momentum);
        assertNumberBetween(deltaWeight2To1, .00415, .00416);
        double deltaWeight3To1 = Node.calculateWeightDelta(learningRate, error1, output3, 0, momentum);
        assertNumberBetween(deltaWeight3To1, .00415, .00416);
        double deltaBias1 = Node.calculateWeightDelta(learningRate, error1, 1, 0, momentum);
        assertNumberBetween(deltaBias1, .005686, .005687);

        double deltaWeightI1To2 = Node.calculateWeightDelta(learningRate, error2, 0, 0, momentum);
        assertEquals(0, deltaWeightI1To2);
        double deltaWeightI2To2 = Node.calculateWeightDelta(learningRate, error2, 0, 0, momentum);
        assertEquals(0, deltaWeightI2To2);
        double deltaBias2 = Node.calculateWeightDelta(learningRate, error2, 1, 0, momentum);
        assertNumberBetween(deltaBias2, .00111, .00112);

        double deltaWeightI1To3 = Node.calculateWeightDelta(learningRate, error3, 0, 0, momentum);
        assertEquals(0, deltaWeightI1To3);
        double deltaWeightI2To3 = Node.calculateWeightDelta(learningRate, error3, 0, 0, momentum);
        assertEquals(0, deltaWeightI2To3);
        double deltaBias3 = Node.calculateWeightDelta(learningRate, error3, 1, 0, momentum);
        assertNumberBetween(deltaBias3, .00111, .00112);

        node1.calcWeightChanges(learningRate, error1, momentum);
        inputWeight2To1 = node1.getInputWeight(0);
        assertNumberBetween(inputWeight2To1, 1.0041, 1.0042);
        inputWeight3To1 = node1.getInputWeight(1);
        assertNumberBetween(inputWeight3To1, 1.0041, 1.0042);
        double bias1 = node1.getBiasWeight();
        assertNumberBetween(bias1, 1.005686, 1.005687);

        node2.calcWeightChanges(learningRate, error2, momentum);
        double inputWeightI1To2 = node2.getInputWeight(0);
        assertEquals(1, inputWeightI1To2);
        double inputWeightI2To2 = node2.getInputWeight(1);
        assertEquals(1, inputWeightI2To2);
        double bias2 = node2.getBiasWeight();
        assertNumberBetween(bias2, 1.00111, 1.00112);

        node3.calcWeightChanges(learningRate, error3, momentum);
        double inputWeightI1To3 = node3.getInputWeight(0);
        assertEquals(1, inputWeightI1To3);
        double inputWeightI2To3 = node3.getInputWeight(1);
        assertEquals(1, inputWeightI2To3);
        double bias3 = node3.getBiasWeight();
        assertNumberBetween(bias3, 1.00111, 1.00112);
    }

    @Test
    void testCalcOutput()
    {
        Node node = Node.ZERO_NODE;
        double output = node.calcOutput(0);
        assertEquals(.5, output);

        output = node.calcOutput(2.5);
        assertNumberBetween(output, .923, .925);

        output = node.calcOutput(-6);
        assertNumberBetween(output, .00246, .00248);

        output = node.calcOutput(5);
        assertNumberBetween(output, .9932, .9934);

        output = node.calcOutput(1);
        assertNumberBetween(output, .73, .7311);

        output = node.calcOutput(2.762);
        assertNumberBetween(output, .9405, .9415);

        output = node.calcOutput(2);
        assertNumberBetween(output, .8805, .8815);
    }

    @Test
    void testCalcGradient()
    {
        Node node = Node.ZERO_NODE;
        double gradient = node.calcGradient(.921);
        assertNumberBetween(gradient, .0727589, .072759);

        gradient = node.calcGradient(.941);
        assertNumberBetween(gradient, .055518, .05552);

        gradient = node.calcGradient(.881);
        assertNumberBetween(gradient, .1048, .1049);

        gradient = node.calcGradient(.731);
        assertNumberBetween(gradient, .1966, .1967);
    }

    @Test
    void testCalcWeightChanges()
    {
        double momentum = 0;
        List<InputWeight> inputWeights = new ArrayList<>(2);
        inputWeights.add(new InputWeight(0, 1));
        inputWeights.add(new InputWeight(0, 1));
        Node node = new Node(inputWeights, 1);
        node.calcWeightChanges(1, .00113, momentum);
        double weight = node.getInputWeight(0);
        assertEquals(1, weight);
        weight = node.getInputWeight(1);
        assertEquals(1, weight);
        weight = node.getBiasWeight();
        assertNumberBetween(weight, 1.00112, 1.00114);

        inputWeights = new ArrayList<>(2);
        inputWeights.add(new InputWeight(.731, 1));
        inputWeights.add(new InputWeight(.731, 1));
        node = new Node(inputWeights, 1);
        node.calcWeightChanges(1, .00575, momentum);
        weight = node.getInputWeight(0);
        assertNumberBetween(weight, 1.00419, 1.00421);
        weight = node.getInputWeight(1);
        assertNumberBetween(weight, 1.00419, 1.00421);
        weight = node.getBiasWeight();
        assertNumberBetween(weight, 1.005749, 1.005751);
    }

    @Test
    void testCalcOutputNodeError()
    {
        Node node = Node.ZERO_NODE;
        double gradient = node.calcGradient(.921);
        double error = node.calcOutputNodeError(gradient, 1, .921);
        assertNumberBetween(error, .00574, .00576);

        gradient = node.calcGradient(.941);
        error = node.calcOutputNodeError(gradient, 0, .941);
        assertNumberBetween(error, -.05225, -.05215);
    }

    @Test
    void testCalcHiddenNodeError()
    {
        Node node = Node.ZERO_NODE;
        double gradient = node.calcGradient(.731);
        List<Double> outputErrors = new ArrayList<>(1);
        outputErrors.add(.00575);
        List<Double> weightsToOutputs = new ArrayList<>(1);
        weightsToOutputs.add(1d);
        double error = node.calcHiddenNodeError(gradient, outputErrors, weightsToOutputs);
        assertNumberBetween(error, .00112, .00114);

        gradient = node.calcGradient(.881);
        outputErrors = new ArrayList<>(1);
        outputErrors.add(-.0522);
        weightsToOutputs = new ArrayList<>(1);
        weightsToOutputs.add(1d);
        error = node.calcHiddenNodeError(gradient, outputErrors, weightsToOutputs);
        assertNumberBetween(error, -.005475, -.005465);
    }

    @Test
    void testCalcNet()
    {
        List<Double> inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(0d);

        Node node = new Node(inputs.size(), 1234);
        double net = node.calcNet(inputs);
        assertNumberBetween(net, .356, .358);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(1d);
        net = node.calcNet(inputs);
        assertNumberBetween(net, .807, .809);

        inputs = new ArrayList<>();
        inputs.add(1d);
        inputs.add(1d);
        net = node.calcNet(inputs);
        assertNumberBetween(net, .955, .956);

        inputs = new ArrayList<>();
        inputs.add(1d);
        inputs.add(0d);
        node = new Node(inputs.size(), 1234);
        net = node.calcNet(inputs);
        assertNumberBetween(net, .504, .505);

        inputs = new ArrayList<>();
        inputs.add(0d);
        inputs.add(1d);
        inputs.add(1d);
        inputs.add(1d);
        inputs.add(0d);
        node = new Node(inputs.size(), 1234);
        net = node.calcNet(inputs);
        assertNumberBetween(net, .471, .472);
    }
}