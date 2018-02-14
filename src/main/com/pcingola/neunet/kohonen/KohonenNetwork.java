package com.pcingola.neunet.kohonen;

import com.pcingola.neunet.Network;

/**
 * A Kohonen neural network
 * 
 * @author pcingola@sinectis.com
 */
public class KohonenNetwork extends Network {

	protected double eta;
	protected int neighbourhood;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public KohonenNetwork(int numberOfLayers) {
		super(numberOfLayers);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	public double getEta() {
		return eta;
	}

	public int getNeighbourhood() {
		return neighbourhood;
	}

	/**
	 * Learning algorithm for a network Note: We assume that this network is
	 * only one layer
	 * 
	 * @param desiredOutputs: not used for unsupervised learning algorithms
	 */
	public void learn(double desiredOutputs[][]) {
		KohonenLayer klay = (KohonenLayer) layer[1];
		klay.setEta(eta);
		klay.setNeighbourhood(neighbourhood);
		klay.calc();
		klay.learn(null);
	}

	public void setEta(double eta) {
		this.eta = eta;
	}

	public void setNeighbourhood(int neighbourhood) {
		this.neighbourhood = neighbourhood;
	}

}