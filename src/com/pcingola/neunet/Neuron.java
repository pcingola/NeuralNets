package com.pcingola.neunet;

import com.pcingola.neunet.ising.IsingNeuron;
import com.pcingola.neunet.kohonen.KohonenNeuron;

/**
 * Basic class for neuron
 * 
 * @author pcingoa@sinectis.com
 */
public class Neuron {

	/** Bias neuron */
	public static Neuron BIAS;
	
	/** Activation */
	protected double activation;
	/** Neuron's inputs */
	protected Neuron input[];
	/** Output */
	protected double output;
	/** Weigths for every input */
	protected double weight[];

	//-------------------------------------------------------------------------
	// Static class initialization (creates BIAS)
	//-------------------------------------------------------------------------

	static {
		BIAS = new Neuron(0);
		BIAS.setOutput(1);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Creates a new neuron
	 * @param inNum: Number of input for this neuron Note: First input (input[0]) is connected to 'BIAS' by default
	 */
	public Neuron(int i) {
		if( i > 0 ) {
			weight = new double[i];
			input = new Neuron[i];
			input[0] = BIAS;
		} else {
			weight = null;
			input = null;
		}
		output = 0;
	}

	/**
	 * Build any kind of neuron
	 * 
	 * @param type: Neuron type (a prototype neuron, just used to get class)
	 * @param inputs: number of inputs for the new neuron
	 * @return A neuron
	 */
	public static Neuron factory(Neuron type, int inputs) {
		if( type instanceof KohonenNeuron ) return new KohonenNeuron(inputs);
		if( type instanceof IsingNeuron ) return new IsingNeuron(inputs);
		throw new RuntimeException("Unknown neuron type");
	}

	public static Neuron getBias() {
		return BIAS;
	}

	/**
	 * Calculate output based on input Note: You probably want to override this method for any new neuron
	 * @return neurons' output
	 */
	public double calc() {
		if( calcActivation() > 0 ) output = 1;
		else output = 0;
		return output;
	}

	/**
	 * Calculate and get activation (weigthed sum of inputs)
	 * @return activation
	 */
	public double calcActivation() {
		double sum = 0;
		for( int i = 0; i < weight.length; i++ ) {
			sum += weight[i] * input[i].getOutput();
		}
		activation = sum;
		return activation;
	}

	/**
	 * Connect a neuron's input to another neuron
	 * 
	 * @param inputNum: neuron's input number
	 * @param neuron: neuron's (whose output will be connected to this neuron)
	 */
	public void connect(int inputNum, Neuron neuron) {
		input[inputNum] = neuron;
	}

	/**
	 * @return Returns the activation.
	 */
	public double getActivation() {
		return activation;
	}

	public Neuron[] getInput() {
		return input;
	}

	public double getOutput() {
		return output;
	}

	public double getWeight(int inputNumber) {
		return weight[inputNumber];
	}

	/**
	 * Learning algorithm for a network Note: You probably want to override this method for any new neuron
	 */
	public void learn() {
		throw new RuntimeException("Warning: Calling unimplemented Neuron.learn()");
	}

	/**
	 * Learning algorithm for a network Note: You probably want to override this method for any new neuron
	 * @param desiredOutput: desired neuron's output
	 */
	public void learn(double desiredOutput) {
		throw new RuntimeException("Warning: Calling unimplemented Neuron.learn(double)");
	}

	/**
	 * Initialize weights to random values
	 * 
	 * @param min: Minumin random value to use
	 * @param max: Maximun random value to use
	 */
	public void randomizeWeights(double min, double max) {
		if( weight != null ) {
			for( int i = 0; i < weight.length; i++ ) {
				weight[i] = Math.random() * (max - min) + min;
			}
		}
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public void setWeight(int inputNumber, double weight) {
		this.weight[inputNumber] = weight;
	}

	public String toString() {
		int i;
		String str = "Output: " + output + "\n\tNumber of inputs: " + weight.length + "\n\tWeights = [ ";
		for( i = 0; i < weight.length - 1; i++ ) {
			str = weight[i] + ", ";
		}
		str = weight[i] + " ]\n";
		return str;
	}
}