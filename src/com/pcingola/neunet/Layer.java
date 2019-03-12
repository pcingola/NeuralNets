package com.pcingola.neunet;

/**
 * Implements a layer of neurons
 * @author pcingola@sinectis.com
 */
public class Layer {

	public static String LAYER_HIDDEN = "Hidden";
	public static String LAYER_INPUT = "Input";
	public static String LAYER_OUTPUT = "Output";

	/** What kind of layer is this? {LAYER_INPUT, LAYER_HIDDEN, LAYER_OUTPUT} */
	protected String layerPossition;
	/** An array of neurons */
	protected Neuron neuron[][];
	/** Layer size */
	protected int sizeX, sizeY;
	/** An empty neuron, just used to know what kind of neurons we are using */
	protected Neuron type;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	/** A default constructor that does nothing */
	protected Layer() {	
	}
	
	/**
	 * A layer of neurons (a 2D array of neurons, i.e. a rectangle)
	 * 
	 * @param type: Neuron type (a neuron used as prototype)
	 * @param sizeX: layer's x size
	 * @param sizeY: layer's x size
	 * @param inputs: Number of inputs per neuron
	 * @param min: Minimun neuron's weight (when initialized randomly)
	 * @param max: Maximun neuron's weight (when initialized randomly)
	 */
	public Layer(Neuron type, int sizeX, int sizeY, int inputs, double min, double max) {
		this.type = type;
		int i, j;
		this.sizeX = sizeX;
		this.sizeY = sizeY;
		neuron = new Neuron[sizeX][sizeY];
		for( i = 0; i < sizeX; i++ ) {
			for( j = 0; j < sizeY; j++ ) {
				neuron[i][j] = Neuron.factory(type, inputs);
				neuron[i][j].randomizeWeights(min, max);
			}
		}
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Calculate the whole layer
	 */
	public void calc() {
		int i, j;
		for( i = 0; i < sizeX; i++ ) {
			for( j = 0; j < sizeY; j++ ) {
				neuron[i][j].calc();
			}
		}
	}

	/**
	 * Connect every neuron in this layer to the preceding layer
	 * @param lay: A layer to connect to.
	 */
	public void connectAll(Layer lay, boolean connectFirsInputToBias) {
		int i, j, li, lj, in;
		Neuron neu;
		for( i = 0; i < getSizeX(); i++ ) {
			for( j = 0; j < getSizeY(); j++ ) {
				neu = getNeuron(i, j);
				in = 0;
				if( connectFirsInputToBias ) {
					neu.connect(0, Neuron.BIAS);
					in = 1;
				}

				for( li = 0; li < lay.getSizeX(); li++ ) { // Start connecting
					// input number 1
					// (input number 0 is
					// bias)
					for( lj = 0; lj < lay.getSizeY(); lj++, in++ ) {
						neu.connect(in, lay.getNeuron(li, lj));
					}
				}
			}
		}
	}

	/**
	 * @return Returns the layerPossition.
	 */
	public String getLayerPossition() {
		return layerPossition;
	}

	public Neuron getNeuron(int i, int j) {
		return neuron[i][j];
	}

	/**
	 * @return Returns the sizeX.
	 */
	public int getSizeX() {
		return sizeX;
	}

	/**
	 * @return Returns the sizeY.
	 */
	public int getSizeY() {
		return sizeY;
	}

	/**
	 * @return Returns the type.
	 */
	public Neuron getType() {
		return type;
	}

	public boolean isHiddenLayer() {
		if( layerPossition != null ) {
			if( layerPossition.equals(LAYER_HIDDEN) ) return true;
		}
		return false;
	}

	public boolean isInputLayer() {
		if( layerPossition != null ) {
			if( layerPossition.equals(LAYER_INPUT) ) return true;
		}
		return false;
	}

	public boolean isOutputLayer() {
		if( layerPossition != null ) {
			if( layerPossition.equals(LAYER_OUTPUT) ) return true;
		}
		return false;
	}

	/**
	 * Learning algorithm for a layer
	 * @param desiredOutputs: an array of desired outputs
	 */
	public void learn(double desiredOutputs[][]) {
		int i, j;
		if( isOutputLayer() ) {
			for( i = 0; i < sizeX; i++ ) {
				for( j = 0; j < sizeY; j++ ) {
					neuron[i][j].learn(desiredOutputs[i][j]);
				}
			}
		} else {
			for( i = 0; i < sizeX; i++ ) {
				for( j = 0; j < sizeY; j++ ) {
					neuron[i][j].learn();
				}
			}
		}
	}

	/**
	 * @param layerPossition: The layerPossition to set.
	 */
	public void setLayerPossition(String layerPossition) {
		this.layerPossition = layerPossition;
	}

	public String toString() {
		return layerPossition + " layer: [" + sizeX + ", " + sizeY + "] type: " + type.getClass().getName();
	}
}