package com.pcingola.neunet.kohonen;

import com.pcingola.neunet.Layer;

/**
 * Implements a layer of kohonen's neurons
 * 
 * @author pcingola@sinectis.com
 */
public class KohonenLayer extends Layer {

	public static KohonenNeuron neuronPrototype = new KohonenNeuron(0);
	
	/** Best neuron (i.e. highest output) */ 
	protected int bestOutputX, bestOutputY;
	/** Learning rate */
	protected double eta;
	/** Neigbourhood size */
	protected int neighbourhood;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public KohonenLayer(int sizeX, int sizeY, int inputs, double min, double max) {
		super(neuronPrototype, sizeX, sizeY, inputs, min, max);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Calculate the whole layer
	 */
	public void calc() {
		double best = -1e10;
		double out;
		int i, j;
		for( i = 0; i < sizeX; i++ ) {
			for( j = 0; j < sizeY; j++ ) {
				out = neuron[i][j].calc();
				if( out > best ) {
					best = out;
					bestOutputX = i;
					bestOutputY = j;
				}
			}
		}
	}

	public int getBestOutputX() {
		return bestOutputX;
	}

	public int getBestOutputY() {
		return bestOutputY;
	}

	public double getEta() {
		return eta;
	}

	public int getNeighbourhood() {
		return neighbourhood;
	}

	/**
	 * Learning algorithm for a layer
	 * @param desiredOutputs an array of desired outputs
	 */
	public void learn(double desiredOutputs[][]) {
		int i, j;
		int ii, jj;
		if( isOutputLayer() ) {
			for( i = -neighbourhood; i < neighbourhood; i++ ) {
				ii = bestOutputX + i;
				if( (ii >= 0) && (ii < sizeX) ) {
					for( j = -neighbourhood; j < neighbourhood; j++ ) {
						jj = bestOutputY + j;
						if( (jj >= 0) && (jj < sizeY) ) neuron[ii][jj].learn(eta);
					}
				}
			}
		}
	}

	public void setEta(double eta) {
		this.eta = eta;
	}

	public void setNeighbourhood(int neighbourhood) {
		this.neighbourhood = neighbourhood;
	}

}