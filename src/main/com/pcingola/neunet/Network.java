package com.pcingola.neunet;

/**
 * A neural network
 * 
 * @author pcingola@sinectis.com
 */
public class Network {

	/** Input patterns [patternNumber][sizeX][sizeY] (input patterns should be same size as inut layer */
	protected double input[][][];
	/** Layers of neurons */
	protected Layer layer[];
	/** Output patterns [patternNumber][sizeX][sizeY] (output patterns should be same size as output layer */
	protected double output[][][];

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	/** Default constructr does nothing! */
	protected Network() {
	}
	
	/** Build lyers */
	public Network(int numberOfLayers) {
		layer = new Layer[numberOfLayers];
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * A layer of neurons (a 2D array of neurons, i.e. a rectangle)
	 * 
	 * @param layerNum: layer number
	 * @param type: Neuron type (a neuron used as prototype)
	 * @param sizeX: layer's x size
	 * @param sizeY: layer's x size
	 * @param inputs: Number of inputs per neuron
	 * @param min: Minimun neuron's weight (when initialized randomly)
	 * @param max: Maximun neuron's weight (when initialized randomly)
	 */
	public void addLayer(int layerNum, Layer lay) {
		layer[layerNum] = lay;
		// Set layer type (related to positon in this network)
		if( layerNum == 0 ) layer[layerNum].setLayerPossition(Layer.LAYER_INPUT);
		else if( layerNum == layer.length - 1 ) layer[layerNum].setLayerPossition(Layer.LAYER_OUTPUT);
		else layer[layerNum].setLayerPossition(Layer.LAYER_HIDDEN);
	}

	/**
	 * Calculate the whole network Note: You probably want to override this method for any new neuron
	 */
	public void calc() {
		for( int l = 1; l <= layer.length; l++ ) { // Do not calculate input layer (start with layer 1)
			layer[l].calc();
		}
	}

	/**
	 * @return Returns the input.
	 */
	public double[][][] getInput() {
		return input;
	}

	/**
	 * @return Returns a layer.
	 */
	public Layer getLayer(int layerNumber) {
		return layer[layerNumber];
	}

	/**
	 * Learning algorithm for a network Note: You probably want to override this method for any new neuron
	 * @param desiredOutputs an array of desired outputs
	 */
	public void learn(double desiredOutputs[][]) {
		for( int l = 0; l <= layer.length; l++ ) {
			layer[l].learn(desiredOutputs);
		}
	}

	/**
	 * @param input: The input to set.
	 */
	public void setInput(double[][][] input) {
		this.input = input;
	}

	/**
	 * Sets input pattern numer 'inputPatternNumber' as input layer (i.e. input layer neuron's output)
	 * @param inputPatternNumber
	 */
	public void setInputPattern(int inputPatternNumber) {
		int sizeX = layer[0].getSizeX();
		int sizeY = layer[0].getSizeY();
		int i, j;
		for( i = 0; i < sizeX; i++ ) {
			for( j = 0; j < sizeY; j++ ) {
				layer[0].getNeuron(i, j).setOutput(input[inputPatternNumber][i][j]);
			}
		}
	}

	public String toString() {
		String str = new String();
		for( int i = 0; i < layer.length; i++ ) {
			str += (layer[i] == null ? "null" : layer[i].toString()) + "\n";
		}
		return str;
	}
}