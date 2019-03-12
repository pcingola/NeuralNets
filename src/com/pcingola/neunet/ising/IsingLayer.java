package com.pcingola.neunet.ising;

import com.pcingola.neunet.Layer;

/**
 * Implements a layer of Ising's neurons
 * 
 * @author pcingola@sinectis.com
 */
public class IsingLayer extends Layer {

	/** In this model, this is the only neuron used (because they are all the same) */
	public static IsingNeuron neuronPrototype = new IsingNeuron(4);

	/** Neuron's outputs */
	protected int outputs[][];
	/** Pseudo-temperature */
	protected double temp;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public IsingLayer(int sizeX, int sizeY) {
		super();
		this.temp = 1.0; // Default temperature
		this.sizeX = sizeX;
		this.sizeY = sizeY;

		// Build array
		outputs = new int[sizeX][sizeY];

		// Random init outputs
		for( int i = 0; i < sizeX; i++ )
			for( int j = 0; j < sizeY; j++ )
				outputs[i][j] = (Math.random() > 0.5 ? 1 : -1);

	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Calculate the whole layer
	 */
	public void calc() {
		int numberOfiterations = sizeX * sizeY;
		int maxX = sizeX - 1;
		int maxY = sizeY - 1;

		for( int i = 0; i < numberOfiterations; i++ ) {
			// Randmoly select a position
			int rx = (int) (Math.random() * sizeX);
			int ry = (int) (Math.random() * sizeY);

			int rxInc = (rx >= maxX ? 0 : rx + 1);
			int rxDec = (rx <= 0 ? maxX : rx - 1);
			int ryInc = (ry >= maxY ? 0 : ry + 1);
			int ryDec = (ry < 1 ? maxY : ry - 1);

			int h = outputs[rxInc][ry] + outputs[rxDec][ry] + outputs[rx][ryInc] + outputs[rx][ryDec];
			outputs[rx][ry] = neuronPrototype.calc(temp, h);
		}
	}

	/**
	 * Ising 'magnetic field'
	 * @return
	 */
	public double field() {
		double sum = 0;
		for( int i = 0; i < sizeX; i++ )
			for( int j = 0; j < sizeY; j++ )
				sum += outputs[i][j];

		return sum / ((double) (sizeX * sizeY));
	}

	public int[][] getOutputs() {
		return outputs;
	}

	public double getTemp() {
		return temp;
	}

	/**
	 * Learning algorithm for a layer
	 * @param desiredOutputs an array of desired outputs
	 */
	public void learn(double desiredOutputs[][]) {
		throw new RuntimeException("Nohing to learn!");
	}

	public void setOutputs(int[][] outputs) {
		this.outputs = outputs;
	}

	public void setTemp(double temp) {
		this.temp = temp;
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();

		for( int i = 0; i < sizeX; i++ ) {
			for( int j = 0; j < sizeY; j++ )
				if( outputs[i][j] > 0 ) sb.append("+");
				else sb.append("-");
			sb.append("\n");
		}

		return sb.toString();
	}
}
